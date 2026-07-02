"""
fit_liver.py
============
Spaceflight-vs-control classification and per-gene attribution for the HEART
application arm (OSD-270, OSD-580, OSD-599).

The classification-as-attribution approach follows Ilangovan et al.
(npj Microgravity, 2024; doi:10.1038/s41526-024-00379-3). This program is an
independent reimplementation of that scaffold: it uses a functional structure,
its own naming, and a SHAP-based attribution layer in place of permutation
importance.

The ONE block reproduced verbatim is the per-study log2 z-score normalisation
(`scale_data`, marked below). It is retained unchanged so the heart arm applies
the identical preprocessing transform used for the liver benchmark, keeping the
two tissues directly comparable. Result-affecting modelling choices kept for
comparability: the five-fold stratified CV seed and the classifier
hyperparameters. Everything else (loader,
matrix assembly, labelling, integration-matrix support, CLI, exports) is new.
"""

import os
import itertools
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
import shap


def scale_data(count_dfs, exclude_test):

    for i in range(len(count_dfs)):
        scaler = StandardScaler()
        df = count_dfs[i]
        if exclude_test:
            test_ids = []
            for k in exclude_test.keys():
                test_ids.append(exclude_test[k]['id']['SF'])
                test_ids.append(exclude_test[k]['id']['GC'])
            test_ids = list(itertools.chain(*test_ids))
            valid_ids = count_dfs[i].index.isin(test_ids)
            train_ids = np.arange(df.index.values.shape[0])[~valid_ids]
            df = df.iloc[train_ids]

        scaler.fit(np.log2(df+1))

        count_dfs[i] = pd.DataFrame(scaler.transform(np.log2(count_dfs[i]+1)), columns=count_dfs[i].columns, index=count_dfs[i].index)
    return count_dfs



# --------------------------------------------------------------------------- #
#  Input discovery and loading  
# --------------------------------------------------------------------------- #
def _dotted_files(parts):
    folder = Path.cwd()
    for part in parts:
        folder = folder / part
    entries = sorted(x for x in folder.iterdir())
    return [x for x in entries if x.name[-4] == '.']


def load_scaled_studies(counts_parts, meta_parts, holdout_nested, do_scale, verbose=False):
    count_paths = _dotted_files(counts_parts)
    meta_paths = _dotted_files(meta_parts)

    meta_stems = {m.name[:-4] for m in meta_paths}
    kept, dropped = [], []
    for c in count_paths:
        (kept if c.name[:-4] in meta_stems else dropped).append(c)
    if verbose and dropped:
        for d in dropped:
            print("Dropping {} (no matching metadata)".format(d.name))
    elif dropped:
        print("Dropped counts without metadata:", ",".join(d.name for d in dropped))

    count_frames = [pd.read_csv(c, index_col=0).transpose() for c in kept]
    meta_frames = [pd.read_csv(m, delimiter="\t") for m in meta_paths]

    if do_scale:
        count_frames = scale_data(count_frames, holdout_nested)

    study_ids = [c.name.split('.')[0] for c in kept]
    return list(zip(study_ids, count_frames, meta_frames))


def keep_valid_treatments(studies, id_col, factor_col, valid_levels):
    pattern = "|".join(valid_levels)
    filtered = []
    for sid, counts, meta in studies:
        wanted = meta.loc[meta[factor_col].str.contains(pattern), id_col]
        wanted = np.intersect1d(counts.index.values, wanted.tolist())
        filtered.append((sid, counts.loc[wanted, :], meta))
    return filtered


# --------------------------------------------------------------------------- #
#  Target labelling  
# --------------------------------------------------------------------------- #
def read_spaceflight_map(meta_parts=('data', 'single_study_metadata'),
                         id_col="Sample Name", factor_col="Factor Value[Spaceflight]"):
    """Build {sample name -> +1 / -1} from the single-study ISA metadata, so
    that studies whose ids carry no 'F' token are not silently mislabelled."""
    base = Path.cwd()
    for part in meta_parts:
        base = base / part
    mapping = {}
    if not base.exists():
        print("WARNING: metadata dir {} not found; relying on the "
              "'F'-in-name fallback only.".format(base))
        return mapping
    for f in sorted(base.glob("*.txt")):
        md = pd.read_csv(f, delimiter="\t")
        if id_col not in md.columns or factor_col not in md.columns:
            continue
        is_sf = md[factor_col].astype(str).str.contains("Space", case=False, na=False)
        for name, sf in zip(md[id_col].astype(str), is_sf):
            mapping[name] = 1 if sf else -1
    return mapping


def assign_targets(expr):
    """Return the {-1, +1} target vector aligned to `expr.index`, using the
    metadata map first and the 'F'-in-name rule only for unmatched samples."""
    sf_map = read_spaceflight_map()
    idx = expr.index.to_series()
    mapped = idx.map(sf_map)
    missing = mapped.isna()
    if missing.any():
        print("WARNING: {} sample(s) lacked a Factor Value[Spaceflight] match; "
              "applying the 'F'-in-name fallback to: {}".format(
                  int(missing.sum()), list(idx[missing][:10])))
        mapped.loc[missing] = np.where(idx[missing].str.contains("F"), 1, -1)
    return mapped.astype(int).values


# --------------------------------------------------------------------------- #
#  Matrix assembly 
# --------------------------------------------------------------------------- #
def standardise_harmonised(expr, prep, holdout_ids):
    X = expr.astype(float)
    if prep == 'count':
        X = np.log2(X + 1)
    train_mask = ~X.index.isin(holdout_ids)
    scaler = StandardScaler().fit(X.loc[train_mask])
    return pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)


def assemble_native(studies):
    expr = pd.concat([counts for _, counts, _ in studies])
    expr = expr.loc[:, expr.isna().sum(axis=0) == 0]
    expr['target'] = assign_targets(expr)
    return expr


def assemble_custom(custom_concat, prep, holdout_ids):
    print("Reading externally harmonised matrix: {}".format(custom_concat))
    expr = pd.read_csv(custom_concat, index_col=0).transpose()
    expr = expr.loc[:, expr.isna().sum(axis=0) == 0]
    if prep in ('count', 'log'):
        expr = standardise_harmonised(expr, prep, holdout_ids)
    expr['target'] = assign_targets(expr)
    return expr


# --------------------------------------------------------------------------- #
#  Estimators and SHAP attribution  
# --------------------------------------------------------------------------- #
def make_estimator(name, params):
    if name == "svm":
        return LinearSVC(**params["svm"])
    if name == "glm":
        return LogisticRegression(**params["glm"])
    if name == "elasticnet":
        return make_pipeline(StandardScaler(), LogisticRegression(**params["elasticnet"]))
    if name == "xgboost":
        return XGBClassifier(**params["xgboost"])
    raise ValueError("Unknown model: {}".format(name))


def terminal_step(model):
    """Final estimator, so coef_/feature_importances_ read uniformly whether or
    not the model is wrapped in a Pipeline (elasticnet)."""
    if hasattr(model, "named_steps"):
        return list(model.named_steps.values())[-1]
    return model


def shap_attribution(model, X_train, X_test, verbose=False):
    """Per-fold SHAP -> (2, p): row 0 signed mean (direction), row 1 mean
    |SHAP| (magnitude, the ranking metric).

    A LinearExplainer must see features in the same space the linear model was
    trained in. The elasticnet pipeline lives in standardised space, so its
    scaler is applied before explaining; passing raw values would rescale each
    gene's attribution by its raw SD and corrupt the ranking. Bare linear
    models (svm/glm) already operate on the pre-scaled matrix."""
    final = terminal_step(model)
    if hasattr(model, "named_steps") and len(model.named_steps) > 1:
        pre = model[:-1]
        Xtr_in = pre.transform(X_train.values)
        Xte_in = pre.transform(X_test.values)
    else:
        Xtr_in = X_train.values
        Xte_in = X_test.values

    try:
        if hasattr(final, "feature_importances_"):          # xgboost
            explainer = shap.TreeExplainer(final)
            sv = explainer.shap_values(Xte_in)
        elif hasattr(final, "coef_"):                       # svm / glm / elasticnet
            background = Xtr_in if Xtr_in.shape[0] <= 100 else \
                Xtr_in[np.random.choice(Xtr_in.shape[0], 100, replace=False)]
            explainer = shap.LinearExplainer(final, background)
            sv = explainer.shap_values(Xte_in)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) == 2 else np.mean(sv, axis=0)
        else:
            if verbose:
                print("  SHAP: unsupported model type; skipping fold.")
            return None
    except Exception as e:
        if verbose:
            print("  SHAP failed for this fold: {}".format(e))
        return None

    if isinstance(sv, list):
        arr = sv[1] if len(sv) == 2 else np.mean(np.stack(sv, axis=0), axis=0)
    else:
        arr = np.asarray(sv)
    if arr.ndim == 3:
        arr = arr[..., 1] if arr.shape[-1] == 2 else arr.mean(axis=-1)

    signed = arr.mean(axis=0)
    magnitude = np.abs(arr).mean(axis=0)
    return np.vstack([signed, magnitude])


# --------------------------------------------------------------------------- #
#  Classification + attribution driver  
# --------------------------------------------------------------------------- #
def classify_and_attribute(expr, requested, holdout_ids=None, use_shap=True, verbose=False):
    params = {
        # svm: dense signed coefficients (GSEA-arm partner to glm)
        "svm": {"penalty": 'l2', "loss": 'squared_hinge', "C": 1.0,
                "class_weight": 'balanced', "random_state": 12345, "max_iter": 50000},
        # glm: L2 (ridge) logistic regression, dense signed coefficients
        "glm": {"penalty": "l2", "C": 1.0, "class_weight": "balanced",
                "max_iter": 10000, "random_state": 0},
        # elasticnet: L1+L2 logistic regression (sparse selector -> ORA arm)
        "elasticnet": {"penalty": "elasticnet", "solver": "saga", "C": 1.0,
                       "l1_ratio": 0.5, "max_iter": 10000, "tol": 1e-4,
                       "random_state": 12345},
        # xgboost: shallow trees, heavy column subsampling, L1/L2 leaf reg
        "xgboost": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3,
                    "subsample": 0.8, "colsample_bytree": 0.3, "reg_alpha": 0.1,
                    "reg_lambda": 1.0, "min_child_weight": 1,
                    "objective": "binary:logistic", "eval_metric": "logloss",
                    "tree_method": "hist", "n_jobs": -1, "random_state": 12345},
    }

    order = ["svm", "glm", "elasticnet", "xgboost"]
    estimators = [(name, make_estimator(name, params)) for name in order if name in requested]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
    X = expr.iloc[:, :-1]
    # target stored as {-1,+1}; re-encode to {0,1} (xgboost rejects {-1,+1};
    # the positive class stays the SF / +1 group).
    y = (expr.iloc[:, -1] > 0).astype(int)

    held_out = bool(holdout_ids)
    if held_out:
        is_test = X.index.isin(holdout_ids)
        train_pos = np.arange(X.shape[0])[~is_test]
        test_pos = np.arange(X.shape[0])[is_test]

    results = {m: {} for m in requested}
    for name, model in estimators:
        splits = zip([train_pos], [test_pos]) if held_out else skf.split(X, y)

        acc_list, coef_list, shap_list = [], [], []
        for train, test in splits:
            model = model.fit(X.iloc[train, :], y.iloc[train])
            acc_list.append(accuracy_score(model.predict(X.iloc[test, :]), y.iloc[test]))

            final = terminal_step(model)
            if hasattr(final, "feature_importances_"):
                coef_list.append(final.feature_importances_)
            else:
                coef_list.append(final.coef_.T)

            if use_shap:
                sv = shap_attribution(model, X.iloc[train], X.iloc[test], verbose=verbose)
                if sv is not None:
                    shap_list.append(sv)

            if verbose:
                print("  [{}] running mean accuracy {:.3f}".format(name, np.mean(acc_list)))

        if use_shap and shap_list:
            signed_stack = np.vstack([s[0] for s in shap_list])
            abs_stack = np.vstack([s[1] for s in shap_list])
            results[name]['shap'] = pd.DataFrame({
                "shap_mean": signed_stack.mean(axis=0),      # direction of effect
                "shap_abs_mean": abs_stack.mean(axis=0),     # magnitude (rank by this)
                "shap_std": signed_stack.std(axis=0),        # cross-fold stability
            }, index=X.columns.values)

        results[name]["accuracy"] = pd.DataFrame(acc_list, columns=["Accuracy"])
        results[name]["model_coefs"] = pd.DataFrame(
            np.mean(coef_list, axis=0), columns=['Score'], index=X.columns.values)

    return results


# --------------------------------------------------------------------------- #
#  Held-out SF + GC pair(s) per liver study 
# --------------------------------------------------------------------------- #
HOLDOUT = {
    "47":      (["Mmus_C57-6T_LVR_FLT_Rep1_F1"],
                ["Mmus_C57-6T_LVR_GC_Rep3_G5"]),
    "168_rr1": (["Mmus_C57-6J_LVR_RR1_FLT_wERCC_Rep1_M25"],
                ["Mmus_C57-6J_LVR_RR1_GC_wERCC_Rep5_M40"]),
    "168_rr3": (["Mmus_BAL-TAL_LVR_RR3_FLT_wERCC_Rep1_F1"],
                ["Mmus_BAL-TAL_LVR_RR3_GC_wERCC_Rep4_G5"]),
    "242":     (["Mmus_C57-6J_LVR_FLT_C1_Rep1_F1"],
                ["Mmus_C57-6J_LVR_GC_C2_Rep1_G1",
                 "Mmus_C57-6J_LVR_CC_C1_Rep1_C1-1",
                 "Mmus_C57-6J_LVR_CC_C2_Rep1_C2-1"]),
    "245":     (["Mmus_C57-6T_LVR_FLT_LAR_Rep1_F1",
                 "Mmus_C57-6T_LVR_FLT_ISS-T_Rep4_F10",
                 "Mmus_C57-6T_LVR_FLT_ISS-T_Rep2_F8"],
                ["Mmus_C57-6T_LVR_GC_ISS-T_Rep1_G4",
                 "Mmus_C57-6T_LVR_GC_ISS-T_Rep2_G9",
                 "Mmus_C57-6T_LVR_GC_LAR_Rep1_G6",
                 "Mmus_C57-6T_LVR_GC_LAR_Rep2_G4"]),
    "379":     (["RR8_LVR_FLT_ISS-T_YNG_FI7", "RR8_LVR_FLT_ISS-T_YNG_FI8",
                 "RR8_LVR_FLT_ISS-T_YNG_FI9", "RR8_LVR_FLT_ISS-T_OLD_FI11",
                 "RR8_LVR_FLT_ISS-T_OLD_FI12", "RR8_LVR_FLT_ISS-T_OLD_FI13"],
                ["RR8_LVR_GC_ISS-T_YNG_GI8", "RR8_LVR_GC_ISS-T_YNG_GI9",
                 "RR8_LVR_GC_ISS-T_YNG_GI10", "RR8_LVR_GC_ISS-T_OLD_GI11",
                 "RR8_LVR_GC_ISS-T_OLD_GI12", "RR8_LVR_GC_ISS-T_OLD_GI13"]),
}


def _nested(holdout):
    """Adapter to the nested {study: {'id': {'SF': [...], 'GC': [...]}}} shape
    that the verbatim `scale_data` consumes."""
    if holdout is None:
        return None
    return {k: {'id': {'SF': list(sf), 'GC': list(gc)}} for k, (sf, gc) in holdout.items()}


def _flat_ids(holdout):
    """Flat list of every held-out sample id (the only thing the standardiser
    and the classifier need)."""
    if holdout is None:
        return []
    return [i for sf, gc in holdout.values() for i in (*sf, *gc)]


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="LIVER spaceflight-vs-control classifier + gene attribution + concat export.")
    parser.add_argument("-C", "--custom_concat", default=None,
                        help="Externally harmonised concatenated matrix (genes x samples). "
                             "If omitted, the native z-score path (load + scale + filter) is used.")
    parser.add_argument("-P", "--prep", default="none", choices=["count", "log", "none"],
                        help="Downstream prep for --custom_concat matrices: "
                             "'count' (log2+global standardise), 'log' (global standardise only), 'none'.")
    parser.add_argument("-T", "--tag", default="",
                        help="Suffix appended to output filenames to keep per-arm results separate.")
    parser.add_argument("-M", "--models", nargs="+", default=["svm", "glm", "elasticnet", "xgboost"],
                        choices=["svm", "glm", "elasticnet", "xgboost"],
                        help="Classifier(s) to fit and attribute.")
    parser.add_argument("--no_test", action="store_true",
                        help="Disable the held-out test set and use 5-fold stratified CV instead.")
    parser.add_argument("--no_shap", action="store_true",
                        help="Skip SHAP attribution (export coefficients/importances + accuracy only).")
    parser.add_argument("-V", "--verbose", action="store_true")
    args = parser.parse_args()

    active = None if args.no_test else HOLDOUT
    holdout_nested = _nested(active)
    holdout_ids = _flat_ids(active)

    # --- build the concatenated matrix ---
    if args.custom_concat is None:                      # native z-score path
        studies = load_scaled_studies(['data', 'norm_counts'],
                                      ['data', 'single_study_metadata'],
                                      holdout_nested, do_scale=True, verbose=args.verbose)
        studies = keep_valid_treatments(
            studies, "Sample Name", "Factor Value[Spaceflight]",
            ['Space Flight', 'Ground Control', 'Cohort Control #1', 'Cohort Control #2'])
        expr = assemble_native(studies)
    else:
        expr = assemble_custom(args.custom_concat, args.prep, holdout_ids)

    print(expr.shape)

    suffix = ('_' + args.tag) if args.tag else ''
    expr.transpose().to_csv("concat_df{}.csv".format(suffix), header=True)

    # --- classify + attribute ---
    results = classify_and_attribute(expr, args.models, holdout_ids=holdout_ids,
                                     use_shap=not args.no_shap, verbose=args.verbose)

    # --- export attributions ---
    os.makedirs("feature_importance", exist_ok=True)
    for m in args.models:
        rd = results.get(m, {})
        if rd.get('model_coefs') is not None:
            coef_name = 'svm_importances' if m == 'svm' else '{}_importances'.format(m)
            rd['model_coefs'].sort_values('Score', ascending=False).to_csv(
                'feature_importance/{}{}.csv'.format(coef_name, suffix), header=True)
        if rd.get('shap') is not None:
            rd['shap'].to_csv('feature_importance/{}_shap_importance{}.csv'.format(m, suffix), header=True)
        if rd.get('accuracy') is not None:
            rd['accuracy'].to_csv('feature_importance/{}_accuracy{}.csv'.format(m, suffix), index=False)

    print("Done.")


if __name__ == "__main__":
    main()
