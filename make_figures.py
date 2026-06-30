#!/usr/bin/env python3
"""
Reproducible figure generation for the spaceflight-integration framework paper.
Builds Fig 2 (framework + recovery), Fig 3 (cardiac programmes),
Fig 4 (divergence->convergence), Fig 5 (cross-arm overlap), and re-creates
Fig 6 (power/permutation/LOSO, recovery axis).

Inputs (all in DATA): merged_vs_single_summary_{liver,heart}.csv,
merged_vs_single_details_heart.csv, between_arms_jaccard_liver.csv,
power_curve.csv, loso.csv, permutation_null.csv, single_study_baselines.csv.
Outputs: fig2..fig6 .pdf/.png in OUT.
"""
import numpy as np, pandas as pd
import os, io
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from PIL import Image
import pdf2image
from matplotlib.lines import Line2D

# --- Paths: relative to this file. Override with FIG_DATA / FIG_OUT env vars. ---
# Inputs live in ./figure_data; PLOS TIFFs are written to ./figures_plos.
_ROOT = Path(__file__).resolve().parent
DATA = os.environ.get("FIG_DATA", str(_ROOT / "figure_data")).rstrip("/\\") + os.sep
OUT  = os.environ.get("FIG_OUT",  str(_ROOT / "figures_plos")).rstrip("/\\") + os.sep
os.makedirs(OUT, exist_ok=True)

# --- Fonts per PLOS: Arial (Times/Symbol also allowed), 8-12 pt. If Arial is
#     not installed matplotlib falls back to DejaVu Sans (visually equivalent). ---
plt.rcParams.update({"font.family":"sans-serif",
                     "font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
                     "font.size":10,"axes.titlesize":11,"axes.titleweight":"bold",
                     "svg.fonttype":"none","pdf.fonttype":42,"ps.fonttype":42})
CDOWN, CUP = "#2c6fbb", "#c0392b"   # down / up
TEAL, GREY = "#16a085", "#7f7f7f"

# =====================================================================
# PLOS Biology figure export
# ---------------------------------------------------------------------
# Specs enforced here:
#   * TIFF, LZW compression, flattened RGB (no alpha, no layers)
#   * 300-600 dpi; max physical width 19.05 cm (7.5 in)
#   * < 10 MB per file (auto-falls back 600 -> 300 dpi if needed)
#   * fonts set to Arial via rcParams above (8-12 pt; see width WARNINGs)
# Each figure is written as Fig<N>.tif in citation order, per the map below.
#
# This script owns 7 of the 9 figures. Fig 2 (PCA, from diagnostics.R) and
# Fig 9 (pooled-vs-integration) are produced outside it -- run those exported
# images through  to_plos_tiff("path/to/img", 2)  and  to_plos_tiff(..., 9).
#
# !! If you apply manuscript edit D5 (move the between-arm-overlap float so it
#    is first cited in order), the numbering shifts: the overlap figure becomes
#    Fig 4 and stability/cardiac/convergence shift to 5/6/7. Update FIG_NUMBER
#    accordingly -- it is the single source of truth for figure numbers.
# =====================================================================
FIG_NUMBER = {
    "fig_batch_correction_diagnostics": 1,   # silhouette trade-off   (Fig 1)
    "fig2_framework_recovery":          3,   # framework + recovery   (Fig 3)
    "fig6_power_permutation_recovery":  4,   # permutation + LOSO     (Fig 4)
    "fig3_cardiac_programmes":          5,   # cardiac programmes     (Fig 5)
    "fig4_divergence_convergence":      6,   # convergence heatmap    (Fig 6)
    "fig5_between_arm_overlap":         7,   # between-arm overlap    (Fig 7)
    "fig7_integration_quality":         8,   # integration quality    (Fig 8)
}

MAX_W_IN      = 7.5          # PLOS max physical width (19.05 cm)
DPI_LADDER    = (600, 300)   # try 600 dpi, fall back to 300 to stay < 10 MB
MAX_BYTES     = 9_500_000
WRITE_PREVIEW = True         # also write a quick .pdf for local \showfigstrue preview

def _normalise_to_plos_tiff(img, fig_number, dpi, label=""):
    img = img.convert("RGB")                          # drop alpha / layers
    w_px, h_px = img.size
    w_in = w_px / dpi
    if w_in > MAX_W_IN + 1e-6:                         # clamp to <= 7.5 in wide
        scale = MAX_W_IN / w_in
        img = img.resize((int(round(w_px * scale)), int(round(h_px * scale))),
                         Image.LANCZOS)
        print(f"  WARNING: Fig{fig_number} ({label}) was designed {w_in:.1f} in "
              f"wide; down-scaled to {MAX_W_IN} in (text ~{scale:.0%} of design "
              f"size). For best print legibility, re-author it at <= {MAX_W_IN} in.")
    out = f"{OUT}Fig{fig_number}.tif"
    img.save(out, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
    return out

def _fig_to_rgb(fig, dpi):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                pad_inches=0.03, facecolor="white")     # white -> opaque, no alpha
    buf.seek(0)
    return Image.open(buf)

def save(fig, name):
    num = FIG_NUMBER.get(name)
    if WRITE_PREVIEW:                                   # legacy-named local preview
        fig.savefig(f"{OUT}{name}.pdf", bbox_inches="tight", facecolor="white")
    if num is None:
        print(f"  NOTE: '{name}' has no PLOS number (Fig 2/9 come from other "
              f"scripts); wrote preview only.")
        plt.close(fig); return
    for dpi in DPI_LADDER:
        out = _normalise_to_plos_tiff(_fig_to_rgb(fig, dpi), num, dpi, name)
        mb = os.path.getsize(out) / 1e6
        if os.path.getsize(out) <= MAX_BYTES:
            print(f"wrote {os.path.basename(out)}  ({name}, {dpi} dpi, {mb:.1f} MB)")
            break
        print(f"  {os.path.basename(out)} is {mb:.1f} MB > 10 MB at {dpi} dpi; "
              f"retrying at lower dpi.")
    plt.close(fig)

def to_plos_tiff(src, fig_number, dpi=600):
    if os.path.splitext(src)[1].lower() == ".pdf":
        from pdf2image import convert_from_path          # needs poppler
        img = convert_from_path(src, dpi=dpi)[0]
    else:
        img = Image.open(src)
    out = _normalise_to_plos_tiff(img, fig_number, dpi, os.path.basename(src))
    print(f"wrote {os.path.basename(out)}  (from {os.path.basename(src)}, {dpi} dpi)")
    return out

#to_plos_tiff('diagnostics_pca.png', 2)



def fig2():
    fig = plt.figure(figsize=(11, 9.4))
    gs = GridSpec(2, 1, height_ratios=[1.08, 1.0], hspace=0.30)

    # ---------------- A: schematic (full width, widened boxes) ----------------
    axA = fig.add_subplot(gs[0]); axA.axis("off")
    axA.set_xlim(0, 13); axA.set_ylim(0, 10)
    axA.set_title("A  Integration--prediction--enrichment framework", loc="left")

    def box(x, y, w, h, txt, fc="#eef3f8", ec="#34495e", fs=9):
        axA.add_patch(FancyBboxPatch((x, y), w, h,
                      boxstyle="round,pad=0.02,rounding_size=0.12",
                      fc=fc, ec=ec, lw=1.2))
        axA.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        axA.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                      mutation_scale=12, lw=1.1, color="#34495e"))

    box(0.2, 7.5, 3.4, 1.9, "Independent RNA-seq studies\n(liver $\\times$6, heart $\\times$3)",
        fc="#fdf2e9")
    box(0.2, 4.7, 3.4, 1.4, "Merge +\nlibrary-size norm.")
    box(4.2, 2.9, 3.1, 5.2, "4 batch-correction\narms\n\nz-score\nDESeq2\nRUVg\nComBat-ref",
        fc="#eafaf1")
    box(7.9, 6.2, 4.9, 2.3, "Dense signed models\n(SVM, ridge-LR)\n$\\rightarrow$ GSEA (GO:BP)",
        fc="#eaf2fb")
    box(7.9, 2.9, 4.9, 2.3, "Sparse selectors\n(elastic net, XGBoost)\n$\\rightarrow$ ORA (multi-source)",
        fc="#fbeeea")
    box(4.2, 0.3, 8.6, 1.7,
        "Comparison metrics: recovery vs single study, between-arm robustness,\n"
        "convergence, complementarity, integration-quality",
        fc="#f4f6f7", fs=8.5)

    arrow(1.9, 7.5, 1.9, 6.1)   # studies -> merge
    arrow(3.6, 5.4, 4.2, 5.4)   # merge -> arms
    arrow(7.3, 6.3, 7.9, 7.0)   # arms -> GSEA
    arrow(7.3, 4.7, 7.9, 4.0)   # arms -> ORA
    arrow(7.95, 6.2, 7.2, 2.0)  # GSEA -> comparison
    arrow(7.95, 2.9, 7.6, 2.0)  # ORA  -> comparison

    # ---------------- B: recovery, all 8 combos, liver + heart ----------------
    axB = fig.add_subplot(gs[1])
    sl = pd.read_csv(DATA+"merged_vs_single_summary_liver.csv")
    sh = pd.read_csv(DATA+"merged_vs_single_summary_heart.csv")
    cells = [("glm","gsea","ridge-LR / GSEA"),("svm","gsea","SVM / GSEA"),
             ("xgboost","ora","XGBoost / ORA"),("elasticnet","ora","elastic net / ORA")]
    def meanrec(df,m,p): 
        s=df[(df.Method==m)&(df.Paradigm==p)]; return s.Pct_Single_and_Merged.mean()
    liver=[meanrec(sl,m,p) for m,p,_ in cells]; heart=[meanrec(sh,m,p) for m,p,_ in cells]
    y=np.arange(len(cells)); h=0.38
    axB.barh(y+h/2, liver, h, color=TEAL, label="liver (validation)")
    axB.barh(y-h/2, heart, h, color="#e67e22", label="heart (application)")
    axB.set_yticks(y); axB.set_yticklabels([c[2] for c in cells]); axB.invert_yaxis()
    axB.set_xlabel("mean single-study recovery (%)")
    axB.set_title("B  Recovery by method $\\times$ paradigm", loc="left")
    for yi,v in zip(y+h/2,liver): axB.text(v+0.5,yi,f"{v:.0f}",va="center",fontsize=8,color=TEAL)
    for yi,v in zip(y-h/2,heart): axB.text(v+0.5,yi,f"{v:.0f}",va="center",fontsize=8,color="#e67e22")
    axB.legend(frameon=False,fontsize=8,loc="lower right"); axB.set_xlim(0,38)
    axB.spines[["top","right"]].set_visible(False)
    save(fig,"fig2_framework_recovery")

# ============ helpers for heart term programmes ============
def load_heart_gsea():
    d = pd.read_csv(DATA+"merged_vs_single_details_heart.csv"); d=d[d.Method!="Method"].copy()
    for c in ["NES.x","p.adjust.x","p.adjust.y"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    d["RR_Mission"]=d["RR_Mission"].astype(str)
    return d[(d.Method.isin(["svm","glm"]))&(d.Paradigm=="gsea")]

PROG = {
 "Immune (down)": r"interferon|antigen|MHC|T cell|lymphocyte|cell killing|cytotox|inflammasome|immun|defense response",
 "ECM / collagen (up)": r"extracellular|collagen|matrix",
 "Metabolic (up)": r"amino acid|fatty acid|carboxylic|organic (an|cat)ion|icosanoid|lipid|hormone metab",
 "Structural / contractile": r"actomyosin|contractile|muscle contraction|action potential|depolar|sarcomere|myofibril|oxidative phosph|calcium-mediated",
}

# ============ FIG 3: cardiac programmes dotplot ============
def fig3():
    g = load_heart_gsea()
    sub = g[g["Sig. Group"].isin(["Both","Merged"])]
    rows=[]
    picks = {
      "Immune (down)": ["cellular response to interferon-beta","response to type II interferon",
                        "antigen processing and presentation","T cell mediated cytotoxicity","cell killing"],
        "Structural / contractile": ["regulation of actomyosin structure organization",
                        "membrane depolarization during cardiac muscle cell action potential"],
      "ECM / collagen (up)": ["extracellular matrix organization","collagen fibril organization",
                        "extracellular matrix assembly"],
      "Metabolic (up)": ["alpha-amino acid metabolic process","carboxylic acid transport",
                        "icosanoid metabolic process"],
    }
    agg = sub.groupby("Description.x").agg(nes=("NES.x","mean"),n=("Arm","size"),
                                           padj=("p.adjust.x","min"))
    for prog,terms in picks.items():
        for t in terms:
            cand=[d for d in agg.index if d.lower().startswith(t.lower()[:20])]
            key=t if t in agg.index else (cand[0] if cand else None)
            if key is None: continue
            r=agg.loc[key]
            rows.append((prog,key,r.nes,r.n,-np.log10(max(r.padj,1e-300))))
    df=pd.DataFrame(rows,columns=["prog","term","nes","n","mlogp"])
    df=df.iloc[::-1].reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(8.6,5.6))
    yt=[]; ylab=[]; colors=[]; y=0; progcol={}
    order=list(picks.keys())[::-1]
    for prog in order:
        block=df[df.prog==prog]
        for _,r in block.iterrows():
            sz=40+ (r.mlogp)*7
            ax.scatter(r.nes,y,s=min(sz,330),color=(CUP if r.nes>0 else CDOWN),
                       edgecolor="k",lw=.4,zorder=3)
            yt.append(y); ylab.append(r.term[:46]); y+=1
        progcol[prog]=(yt[-len(block)], yt[-1])
        y+=0.8
    ax.axvline(0,color="grey",lw=.8,ls="--")
    ax.set_yticks(yt); ax.set_yticklabels(ylab,fontsize=8)
    ax.set_xlabel("mean merged NES  (<0 down / >0 up under spaceflight)")
    ax.set_title("Integrated cardiac signature: coordinated programmes",loc="left")
    # programme labels on right
    for prog,(y0,y1) in progcol.items():
        ax.text(ax.get_xlim()[1]*0.98,(y0+y1)/2,prog,rotation=270,va="center",ha="left",
                fontsize=8.5,fontweight="bold",color="#34495e")
    ax.set_xlim(-3.0,2.6)
    from matplotlib.lines import Line2D
    leg=[Line2D([0],[0],marker="o",color="w",markerfacecolor=CDOWN,markersize=9,label="down"),
         Line2D([0],[0],marker="o",color="w",markerfacecolor=CUP,markersize=9,label="up"),
         Line2D([0],[0],marker="o",color="w",markerfacecolor="grey",markersize=6,label="small $-\\log_{10}p$"),
         Line2D([0],[0],marker="o",color="w",markerfacecolor="grey",markersize=12,label="large $-\\log_{10}p$")]
    ax.legend(handles=leg,frameon=False,fontsize=8,loc="lower left",ncol=2)
    ax.spines[["top","right"]].set_visible(False)
    save(fig,"fig3_cardiac_programmes")

# ============ FIG 4: divergence -> convergence heatmap ============
def fig4():
    g = load_heart_gsea()
    progs=list(PROG.keys())
    sources=["OSD-270","OSD-580","OSD-599","Integrated"]
    M=np.full((len(progs),len(sources)),np.nan)
    for i,prog in enumerate(progs):
        sel=g[g["Description.x"].str.contains(PROG[prog],case=False,na=False)]
        for j,mission in enumerate(["270","580","599"]):
            sm=sel[sel.RR_Mission==mission]
            if len(sm): M[i,j]=-np.log10(max(sm["p.adjust.y"].min(),1e-300))
        if len(sel): M[i,3]=-np.log10(max(sel["p.adjust.x"].min(),1e-300))
    M=np.clip(M,0,30)
    fig,ax=plt.subplots(figsize=(6.4,4.2))
    im=ax.imshow(M,cmap="rocket_r" if False else "viridis",aspect="auto",vmin=0,vmax=30)
    ax.set_xticks(range(len(sources))); ax.set_xticklabels(sources)
    ax.set_yticks(range(len(progs))); ax.set_yticklabels(progs)
    for i in range(len(progs)):
        for j in range(len(sources)):
            if not np.isnan(M[i,j]):
                ax.text(j,i,f"{M[i,j]:.0f}",ha="center",va="center",
                        color="white" if M[i,j]>15 else "black",fontsize=9)
    ax.axvline(2.5,color="white",lw=2)
    ax.set_title("Divergent single studies $\\rightarrow$ convergent integrated signature",loc="left",fontsize=10.5)
    cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04); cb.set_label("$-\\log_{10}$ adj. $p$ (best term)")
    save(fig,"fig4_divergence_convergence")

# ============ FIG 5: cross-arm overlap (batch robustness) ============
def fig5():
    j=pd.read_csv(DATA+"between_arms_jaccard_liver.csv")
    arms=["zscore","deseq2","ruvg","combat_ref"]
    cells=[("glm","gsea","ridge-LR / GSEA"),("svm","gsea","SVM / GSEA"),
           ("xgboost","ora","XGBoost / ORA"),("elasticnet","ora","elastic net / ORA")]
    fig, axes = plt.subplots(
    1, 4,
    figsize=(13, 3.6),
    gridspec_kw={"wspace": 0.5}
    )
    for ax,(m,p,title) in zip(axes,cells):
        sub=j[(j.Method==m)&(j.Paradigm==p)]
        M=np.full((4,4),np.nan)
        for _,r in sub.iterrows():
            if r.arm_a in arms and r.arm_b in arms:
                M[arms.index(r.arm_a),arms.index(r.arm_b)]=r.jaccard
        im=ax.imshow(M,cmap="magma",vmin=0,vmax=1,aspect="equal")
        ax.set_xticks(range(4)); ax.set_xticklabels(arms,rotation=45,ha="right",fontsize=8)
        ax.set_yticks(range(4)); ax.set_yticklabels(arms,fontsize=8)
        off=M[~np.eye(4,dtype=bool)]; mean_off=np.nanmean(off)
        ax.set_title(f"{title}\nmean off-diag = {mean_off:.2f}",fontsize=9.5)
        for a in range(4):
            for b in range(4):
                if not np.isnan(M[a,b]):
                    ax.text(b,a,f"{M[a,b]:.2f}",ha="center",va="center",fontsize=7,
                            color="white" if M[a,b]<0.5 else "black")
    cb=fig.colorbar(im,ax=axes,fraction=0.012,pad=0.01); cb.set_label("Jaccard (term overlap)")
    fig.suptitle("Between-arm reproducibility of enriched terms (liver)",x=0.07,ha="left",fontweight="bold")
    save(fig,"fig5_between_arm_overlap")


    # ============ FIG 6: stability (LOSO) + label-specificity (permutation) ============
def fig6():
    lo=pd.read_csv(DATA+"loso.csv"); pm=pd.read_csv(DATA+"permutation_null.csv")
    obs=pm[pm.replicate==0].iloc[0]; null=pm[pm.replicate>0]
    namemap={"47":"GLDS-47","168_rr1":"GLDS-168 RR1","168_rr3":"GLDS-168 RR3",
             "242":"GLDS-242 (RR-9, male)","245":"GLDS-245","379":"GLDS-379"}
    fig=plt.figure(figsize=(11,4.0)); gs=GridSpec(1,2,wspace=0.30)
    axA=fig.add_subplot(gs[0,1]); axA.hist(null["recovery"],bins=30,color=GREY,alpha=.65)
    axA.axvline(obs["recovery"],color="#d62728",lw=2)
    axA.text(obs["recovery"]-2,axA.get_ylim()[1]*0.6,f"observed\n{obs['recovery']:.1f}%",
             color="#d62728",ha="right",fontsize=9)
    axA.set_xlim(-2,102)
    axA.set_xlabel("recovery under shuffled labels (%)"); axA.set_ylabel("permutations")
    axA.set_title("B  Label-permutation null ($p=0.005$)",loc="left")
    
    axB=fig.add_subplot(gs[0,0]); lo2=lo.sort_values("recovery")
    labels=[namemap.get(str(s),str(s)) for s in lo2["left_out"]]
    axB.barh(labels,lo2["recovery"],color=CDOWN); axB.set_xlim(0,100)
    axB.axvline(100,color=GREY,ls="--",lw=1)
    axB.set_xlabel("recovery of full-data signature (%)"); axB.set_ylabel("study removed")
    axB.set_title("A  Leave-one-study-out",loc="left")
    for i,(_,r) in enumerate(lo2.iterrows()):
        axB.text(r["recovery"]+1.5,i,f"{r['recovery']:.0f}",va="center",fontsize=8,color=GREY)
    save(fig,"fig6_stability")
# ============ FIG (new): batch-correction diagnostics ============
def fig_diagnostics():
    d = pd.read_csv('path/to/file' + "diagnostics_metrics.csv")
    d = d.sort_values("score").reset_index(drop=True)

    fig = plt.figure(figsize=(11, 4.3))
    gs = GridSpec(1, 2, width_ratios=[1.18, 1], wspace=0.30)

    # --- A: batch-vs-biology trade-off ---
    axA = fig.add_subplot(gs[0])
    xpad = 0.04
    xlo, xhi = d.sil_by_study.min() - xpad, d.sil_by_study.max() + xpad
    # faint iso-score guides: sil_condition = sil_study + score
    xs = np.array([xlo, xhi])
    for sval in (-0.2, -0.1, 0.0, 0.1, 0.2):
        axA.plot(xs, xs + sval, lw=0.7, zorder=0,
                 ls="-" if sval == 0 else ":",
                 color="#34495e" if sval == 0 else "#cdd5dd")
    # shade favourable region (low study sep, high condition sep = top-left)
    axA.axvspan(xlo, 0, color=TEAL, alpha=0.05, zorder=0)

    # per-method label offsets so close points (RUVg / ComBat-ref) don't collide
    off = {"DESeq2": (6, 6), "zscore": (6, 8), "RUVg": (8, 9),
           "ComBat-ref": (8, -12), "Raw (log2-CPM)": (-8, 8)}
    for _, r in d.iterrows():
        c = TEAL if r.score > 0 else CUP
        axA.scatter(r.sil_by_study, r.sil_by_condition, s=70,
                    color=c, edgecolor="k", lw=0.5, zorder=3)
        dx, dy = off.get(r.method, (6, 6))
        axA.annotate(r.method, (r.sil_by_study, r.sil_by_condition),
                     textcoords="offset points", xytext=(dx, dy),
                     ha="right" if dx < 0 else "left", fontsize=8.5)

    axA.axhline(0, color="grey", lw=0.6, ls="--", zorder=1)
    axA.axvline(0, color="grey", lw=0.6, ls="--", zorder=1)
    axA.set_xlim(xlo, xhi)
    axA.set_xlabel("silhouette by study  ($\\leftarrow$ less residual batch structure)")
    axA.set_ylabel("silhouette by condition\n(biology preserved $\\rightarrow$)")
    axA.set_title("A  Batch removal vs biology preservation", loc="left")
    axA.annotate("ideal", (xlo + 0.01, d.sil_by_condition.max()),
                 fontsize=8.5, fontweight="bold", color=TEAL, va="top")
    axA.spines[["top", "right"]].set_visible(False)

    # --- B: composite score ranking ---
    axB = fig.add_subplot(gs[1])
    y = np.arange(len(d))
    colors = [TEAL if s > 0 else CUP for s in d.score]
    axB.barh(y, d.score, color=colors, edgecolor="k", lw=0.4)
    axB.axvline(0, color="grey", lw=0.8)
    axB.set_yticks(y)
    axB.set_yticklabels(d.method)
    axB.set_xlabel("integration score  (sil$_{condition}$ $-$ sil$_{study}$)")
    axB.set_title("B  Composite ranking (higher = better)", loc="left")
    for yi, v in zip(y, d.score):
        axB.text(v + (0.006 if v >= 0 else -0.006), yi, f"{v:+.3f}",
                 va="center", ha="left" if v >= 0 else "right", fontsize=8)
    axB.set_xlim(d.score.min() - 0.06, d.score.max() + 0.06)
    axB.spines[["top", "right"]].set_visible(False)

    save(fig, "fig_batch_correction_diagnostics")




    # ============ FIG 7: integration quality (Desiderata 1 & 2, liver) ============
def fig7_integration_quality():
    from matplotlib.lines import Line2D
    ARM = {"zscore":"#2c6fbb","deseq2":"#16a085","ruvg":"#e67e22","combat_ref":"#8e44ad"}
    ARM_LAB = {"zscore":"z-score","deseq2":"DESeq2","ruvg":"RUVg","combat_ref":"ComBat-ref"}
    arms = ["zscore","deseq2","ruvg","combat_ref"]

    mech = pd.read_csv(DATA+"d1_retention_summary.csv")
    deno = pd.read_csv(DATA+"d1_denoising_loso_meta.csv")
    rel  = pd.read_csv(DATA+"d1_singleonly_reliability.csv")
    verd = pd.read_csv(DATA+"d2_merged_only_verdict.csv")
    disc = pd.read_csv(DATA+"d2_discovery_summary.csv")

    fig = plt.figure(figsize=(11.4,8.6))
    gs = GridSpec(2,2, hspace=0.34, wspace=0.26)
    LIM = (0.30,0.82)

    # --- A: D1-MECHANICAL -- full-merge retention (confounded by construction) ---
    axA = fig.add_subplot(gs[0,0])
    g = mech[(mech.Paradigm=="gsea") & mech.AUC_retain_by_signif.notna() & (mech.Method=="svm")].copy()
    g["is_rr1"] = g.RR_Mission.eq("RR1_NASA")
    #axA.plot(LIM, LIM, ls=":", lw=0.8, color="#b9c2cb", zorder=0)
    axA.axvline(0.5, ls="--", lw=0.8, color="grey"); axA.axhline(0.5, ls="--", lw=0.8, color="grey")
    for arm in arms:
        s = g[g.Arm==arm]
        for is_rr1,sub in s.groupby("is_rr1"):
            axA.scatter(sub.AUC_retain_by_signif, sub.AUC_retain_by_absNES, s=46,
                        facecolor= ARM[arm],
                        edgecolor=ARM[arm], lw=1.3 if is_rr1 else 0.5, zorder=3)
    axA.set_xlim(*LIM); axA.set_ylim(*LIM)
    axA.set_xlabel("AUC ranked by single-study significance\n(retained vs dropped)")
    axA.set_ylabel("AUC ranked by\neffect size ($|\\mathrm{NES}|$)")
    axA.set_title("A  Retention is a significance filter\n(mega-analysis; confounded by construction)", loc="left", fontsize=10)

    leg = [Line2D([0],[0],marker="o",ls="",mfc=ARM[a],mec=ARM[a],ms=7,label=ARM_LAB[a]) for a in arms]

    axA.legend(handles=leg, frameon=False, fontsize=7.5, loc="upper left", ncol=1, handletextpad=0.3)
    axA.spines[["top","right"]].set_visible(False)

    # --- B: D1-DENOISING -- confound-free leave-one-study-out meta-GSEA ---
    axB = fig.add_subplot(gs[0,1])
    #axB.plot(LIM, LIM, ls=":", lw=0.8, color="#b9c2cb", zorder=0)
    axB.axvline(0.5, ls="--", lw=0.8, color="grey"); axB.axhline(0.5, ls="--", lw=0.8, color="grey")
    lab = {"RR1_NASA":"RR-1","RR6":"RR-6","RR8":"RR-8","RR9":"RR-9","RR3":"RR-3","RR47":"RR-47"}
    loff = {"RR1_NASA":(7,5),"RR6":(-10,7),"RR8":(8,-12),"RR9":(8,-11),"RR3":(9,-2),"RR47":(8,5)}
    for _,r in deno.iterrows():
        sig = r.perm_p < 0.05
        axB.scatter(r.AUC_by_signif, r.AUC_by_absNES, s=70+260*r.recovered_frac,
                    facecolor=CDOWN if sig else "none", edgecolor=CDOWN, lw=1.4, alpha=0.85, zorder=3)
        dx,dy = loff.get(r.RR_Mission,(7,5))
        axB.annotate(lab.get(r.RR_Mission,r.RR_Mission), (r.AUC_by_signif,r.AUC_by_absNES),
                     textcoords="offset points", xytext=(dx,dy),
                     ha="right" if dx<0 else "left", fontsize=8, color="#34495e")
    axB.set_xlim(*LIM); axB.set_ylim(*LIM)
    axB.set_xlabel("AUC ranked by single-study significance\n(recovered vs not, $k$-excluded meta-reference)")
    axB.set_ylabel("AUC ranked by\neffect size ($|\\mathrm{NES}|$)")
    axB.set_title("B  ...and the filter survives confound removal\n(per study; leave-one-study-out meta)", loc="left", fontsize=10)
    leg2 = [Line2D([0],[0],marker="o",ls="",mfc=CDOWN,mec=CDOWN,ms=8,label="beats permutation null"),
            Line2D([0],[0],marker="o",ls="",mfc="none",mec=CDOWN,ms=8,mew=1.4,label="not significant (RR-3)"),
            Line2D([0],[0],marker="o",ls="",mfc=GREY,mec=GREY,ms=5,label="small / large recovered fraction"),
            Line2D([0],[0],marker="o",ls="",mfc=GREY,mec=GREY,ms=11,label="")]
    axB.legend(handles=leg2, frameon=False, fontsize=7.5, loc="upper left", handletextpad=0.3, labelspacing=0.4)
    axB.spines[["top","right"]].set_visible(False)

    # --- C: D1-RELIABILITY -- reproducibility vs source-study significance ---
    axC = fig.add_subplot(gs[1,0])
    rel["padj_full"] = 10.0**(-rel.single_logp)                  # full-study p.adjust
    rel["reproducible"] = rel.within_study_selection_freq >= 0.6 # = STABLE_FREQ (no mid mass)
    bands = [(0.0,0.001,"$\\leq$0.001"),(0.001,0.01,"0.001-\n0.01"),
             (0.01,0.05,"0.01-\n0.05"),(0.05,0.10,"0.05-\n0.10")]
    x = np.arange(len(bands)); rep=[]; notr=[]; fr=[]
    for lo,hi,_ in bands:
        m = (rel.padj_full>lo) & (rel.padj_full<=hi)
        r = int((m & rel.reproducible).sum()); nr = int((m & ~rel.reproducible).sum())
        rep.append(r); notr.append(nr); fr.append(r/max(1,r+nr))
    rep=np.array(rep); notr=np.array(notr); tot=rep+notr
    axC.bar(x, rep,  color=TEAL,      label="reproducible ($\\geq$60% of subsamples)")
    axC.bar(x, notr, bottom=rep, color="#cfd4d8", label="not reproducible")
    for xi,t,f in zip(x,tot,fr):
        axC.text(xi, t+5, f"{f*100:.0f}%", ha="center", fontsize=9, fontweight="bold", color="#34495e")
    axC.set_xticks(x); axC.set_xticklabels([b[2] for b in bands])
    axC.set_xlabel("significance in the source study (full-study $p_{\\mathrm{adj}}$)")
    axC.set_ylabel(f"single-only terms (count, n = {tot.sum()})")
    axC.set_ylim(0, tot.max()*1.30)
    axC.set_title("C  Dropped terms are reproducible (71% overall) unless\nthey were marginal in the source study", loc="left", fontsize=10)
    axC.legend(frameon=False, fontsize=7.8, loc="upper left", handletextpad=0.5)
    axC.annotate("", xy=(3.35,-0.20), xytext=(-0.35,-0.20), xycoords=("data","axes fraction"),
                 arrowprops=dict(arrowstyle="->", color="#9aa3ab", lw=1.0), annotation_clip=False)
   # axC.text(1.5,-0.255,"stronger $\\rightarrow$ weaker", transform=axC.get_xaxis_transform(),
   #          ha="center", va="top", fontsize=7.8, color="#9aa3ab")
   # axC.text(0.02, 0.40, "strong terms: ~all\nsurvive subsampling", transform=axC.transAxes,
   #          fontsize=7.8, color=TEAL, va="top")
   # axC.annotate("near-threshold: non-recovery is\npower loss, not noise",
   #              xy=(3, rep[3]+notr[3]*0.5), xytext=(2.05, tot.max()*1.02),
   #              fontsize=7.8, color=CUP, ha="left", va="top",
   #              arrowprops=dict(arrowstyle="->", color=CUP, lw=0.9))
    axC.spines[["top","right"]].set_visible(False)

    # --- D: D2 -- merged-only verdict by correction arm ---
    axD = fig.add_subplot(gs[1,1])
    VC = {"credible_discovery":"#16a085","fragile":"#e0a32e","artifact_candidate":"#bdc3c7"}
    x = np.arange(len(arms))
    cred = np.array([(verd[verd.Arm==a].verdict=="credible_discovery").sum() for a in arms])
    frag = np.array([(verd[verd.Arm==a].verdict=="fragile").sum()            for a in arms])
    arte = np.array([(verd[verd.Arm==a].verdict=="artifact_candidate").sum() for a in arms])
    axD.bar(x, cred, color=VC["credible_discovery"], label="credible")
    axD.bar(x, frag, bottom=cred, color=VC["fragile"], label="fragile")
    axD.bar(x, arte, bottom=cred+frag, color=VC["artifact_candidate"], label="candidate artefact")
    tot = cred+frag+arte
    for xi,a,t in zip(x,arms,tot):
        d = disc[disc.Arm==a]; frac = d.n_meta_recovered.sum()/d.n_merged_only.sum()
        axD.text(xi, t+6, f"{frac*100:.0f}%", ha="center", fontsize=9, fontweight="bold", color="#34495e")
    axD.set_xticks(x); axD.set_xticklabels([ARM_LAB[a] for a in arms])
    axD.set_ylabel("merged-only terms (count)")
    axD.set_ylim(0, tot.max()*1.34)
    axD.set_title("D  Merged-only terms are largely corroborated\n(% = meta-corroborated fraction per arm)", loc="left", fontsize=10)
    axD.legend(frameon=False, fontsize=8, loc="upper center",
               bbox_to_anchor=(0.5,0.99), ncol=3, columnspacing=1.1, handletextpad=0.4)
    axD.spines[["top","right"]].set_visible(False)

    save(fig, "fig7_integration_quality")


plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "svg.fonttype": "none", "pdf.fonttype": 42})
TEAL, ORANGE, GREY = "#16a085", "#e67e22", "#7f7f7f"          # liver / heart / neutral
TCOL = {"liver": TEAL, "heart": ORANGE}
BASE = "model_free_de"
DENSE = ["svm", "glm"]                                         # GSEA-route classifiers
PRETTY = {"model_free_de": "pooled baseline (DE)"}

def save(fig, name):
    fig.savefig(OUT + name + ".pdf", bbox_inches="tight")
    fig.savefig(OUT + name + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig); print("wrote", name)

def cell_label(arm, method):
    return f"{arm}+{method}".replace("combat_ref", "ComBat-ref").replace("zscore", "z-score")

# ---------------------------------------------------------------- load
def load():
    rec, ref, ver = {}, {}, {}
    for t in ("liver", "heart"):
        r = pd.read_csv(DATA + f"{t}_recovery_leaderboard.csv")
        r = r[(r.Paradigm == "gsea") & ((r.Method.isin(DENSE)) | (r.Approach == BASE))].copy()
        rec[t] = r
        ref[t] = pd.read_csv(DATA + f"{t}_pooled_vs_referee_summary.csv")
        ver[t] = pd.read_csv(DATA + f"{t}_pooled_merged_only_verdict.csv")
    return rec, ref, ver

# ---------------------------------------------------------------- figure
def fig_pooled_vs_integration():
    rec, ref, ver = load()

    # row order: baseline on top, then dense grid cells by mean recovery across tissues
    grid = (rec["liver"].merge(rec["heart"], on=["Approach", "Arm", "Method"],
                               suffixes=("_L", "_H")))
    grid = grid[grid.Approach != BASE].copy()
    grid["mr_L"] = grid.mean_recovery_perstudy_L * 100
    grid["mr_H"] = grid.mean_recovery_perstudy_H * 100
    grid["ord"] = (grid.mr_L + grid.mr_H) / 2
    grid = grid.sort_values("ord", ascending=False).reset_index(drop=True)

    rows = ["pooled baseline (DE)"] + [cell_label(a, m) for a, m in zip(grid.Arm, grid.Method)]
    bL = [rec["liver"].loc[rec["liver"].Approach == BASE, "mean_recovery_perstudy"].iloc[0] * 100] + list(grid.mr_L)
    bH = [rec["heart"].loc[rec["heart"].Approach == BASE, "mean_recovery_perstudy"].iloc[0] * 100] + list(grid.mr_H)

    fig = plt.figure(figsize=(12, 8.6))
    gs = GridSpec(2, 2, height_ratios=[1.12, 1.0], hspace=0.34, wspace=0.26)

    # =========================================================== A: recovery
    axA = fig.add_subplot(gs[0, :])
    y = np.arange(len(rows)); h = 0.4
    axA.axhspan(-0.5, 0.5, color="#000000", alpha=0.06, zorder=0)      # highlight baseline row
    axA.axhline(0.5, color="#aaaaaa", lw=0.9, ls="--", zorder=1)
    axA.barh(y - h/2, bL, h, color=TEAL,   zorder=3, label="liver (validation, $n{=}137$)",
             edgecolor="k", linewidth=[1.1] + [0]*(len(rows)-1))
    axA.barh(y + h/2, bH, h, color=ORANGE, zorder=3, label="heart (application, $n{=}51$)",
             edgecolor="k", linewidth=[1.1] + [0]*(len(rows)-1))
    for yi, v in zip(y - h/2, bL):
        axA.text(v + 0.6, yi, f"{v:.1f}", va="center", fontsize=8, color=TEAL)
    for yi, v in zip(y + h/2, bH):
        axA.text(v + 0.6, yi, f"{v:.1f}", va="center", fontsize=8, color="#cf711f")
    axA.set_yticks(y); axA.set_yticklabels(rows); axA.invert_yaxis()
    axA.get_yticklabels()[0].set_fontweight("bold")
    axA.set_xlim(0, 76)
    axA.set_xlabel("mean single-study recovery (%)")
    axA.set_title("A  Single-study recovery: pooled baseline vs the integration grid (GSEA route)",
                  loc="left")
    axA.text(0.5, len(rows)-0.4, "sparse (elastic-net / gradient-boosted) cells omitted: "
             "0 GSEA terms by construction", fontsize=7.5, color=GREY, style="italic")
    axA.legend(frameon=False, fontsize=8.5, loc="lower right")
    axA.spines[["top", "right"]].set_visible(False)

    # =========================================================== B: referee Pareto
    axB = fig.add_subplot(gs[1, 0])
    smax = max(ref["liver"].n_meta_recovered.max(), ref["heart"].n_meta_recovered.max())
    def msize(n): return 30 + 470 * (n / smax)
    base_off = {"liver": (-8, 11, "right"), "heart": (11, -1, "left")}
    cr_off   = {"liver": (4, -15, "left"), "heart": (-8, -13, "right")}
    for t in ("liver", "heart"):
        d = ref[t]
        cls = d[d.Approach != BASE]
        axB.scatter(cls.frac_all_sig_referee_backed, cls.frac_recovered,
                    s=msize(cls.n_meta_recovered), color=TCOL[t], alpha=0.55,
                    edgecolor="k", lw=0.4, zorder=3)
        b = d[d.Approach == BASE].iloc[0]
        axB.scatter(b.frac_all_sig_referee_backed, b.frac_recovered,
                    s=msize(b.n_meta_recovered), color=TCOL[t], marker="*",
                    edgecolor="k", lw=1.0, zorder=5)
        dx, dy, ha = base_off[t]
        axB.annotate(f"pooled baseline ({t})",
                     (b.frac_all_sig_referee_backed, b.frac_recovered),
                     textcoords="offset points", xytext=(dx, dy), fontsize=8,
                     fontweight="bold", color=TCOL[t], va="center", ha=ha)
        cr = cls.loc[cls.Approach == "combat_ref__glm"].iloc[0]
        cdx, cdy, cha = cr_off[t]
        axB.annotate("ComBat-ref+glm", (cr.frac_all_sig_referee_backed, cr.frac_recovered),
                     textcoords="offset points", xytext=(cdx, cdy), fontsize=7.5,
                     color=TCOL[t], ha=cha)
    axB.set_xlabel("precision  (referee-backed fraction of all significant terms)")
    axB.set_ylabel("recall\n(referee-corroborated fraction\nof merged-only terms)")
    axB.set_title("B  Agreement with the meta-analytic referee", loc="left")
    axB.set_xlim(0.22, 0.88); axB.set_ylim(0.18, 0.82)

    sleg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY, markeredgecolor="k",
                   markersize=np.sqrt(msize(n))/np.sqrt(np.pi)/2.2, label=f"{n}")
            for n in (50, 200, 500)]
    l1 = axB.legend(handles=sleg, title="corroborated\nterms", frameon=False, fontsize=7.5,
                    title_fontsize=7.5, loc="lower right", labelspacing=1.3, borderpad=1.0)
    axB.add_artist(l1)
    axB.legend(handles=[Line2D([0], [0], marker="*", color="w", markerfacecolor=GREY,
                               markeredgecolor="k", markersize=12, label="pooled baseline"),
                        Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY,
                               markeredgecolor="k", markersize=8, label="integration cell")],
               frameon=False, fontsize=7.5, loc="upper left")
    axB.spines[["top", "right"]].set_visible(False)

    # =========================================================== C: corroborated yield
    axC = fig.add_subplot(gs[1, 1])
    yt, ylab, yc = [], [], 0
    for t in ("liver", "heart"):
        v = ver[t]; v = v[v.Approach == BASE]
        ncred = int((v.verdict == "credible_discovery").sum())
        nfrag = int((v.verdict == "fragile").sum())
        rr = ref[t]
        cls_best = rr.loc[rr.Approach == "combat_ref__glm"].iloc[0]
        ncls = int(cls_best.n_meta_recovered)
        c = TCOL[t]
        # baseline bar (credible solid + fragile lighter)
        axC.barh(yc, ncred, color=c, edgecolor="k", lw=0.4, zorder=3)
        axC.barh(yc, nfrag, left=ncred, color=c, alpha=0.40, edgecolor="k", lw=0.4, zorder=3)
        axC.text(ncred/2, yc, f"{ncred}\ncredible", ha="center", va="center",
                 fontsize=7.5, color="white", fontweight="bold")
        axC.text(ncred + nfrag + 6, yc, f"{ncred+nfrag} corroborated", va="center", fontsize=8,
                 color=c, fontweight="bold")
        yt.append(yc); ylab.append(f"{t}\npooled baseline"); yc += 1
        # best classifier bar
        axC.barh(yc, ncls, color=GREY, edgecolor="k", lw=0.4, zorder=3)
        axC.text(ncls + 6, yc, f"{ncls}", va="center", fontsize=8, color=GREY)
        yt.append(yc); ylab.append("best classifier\n(ComBat-ref+glm)"); yc += 1.4
    axC.set_yticks(yt); axC.set_yticklabels(ylab, fontsize=8.5); axC.invert_yaxis()
    axC.set_xlim(0, 660)
    axC.set_xlabel("referee-corroborated merged-only terms (count)")
    axC.set_title("C  Corroborated cross-study discoveries", loc="left")
    axC.legend(handles=[
        Line2D([1], [0], marker="s", color="w", markerfacecolor=GREY, markersize=9,
               label="credible (recovered + LOSO-robust + concordant)"),
        Line2D([1], [0], marker="s", color="w", markerfacecolor=GREY, markersize=9,
               alpha=0.45, label="fragile (recovered, not LOSO-robust)")],
        frameon=False, fontsize=7.3, loc="lower right")
    axC.spines[["top", "right"]].set_visible(False)

    save(fig, "fig_pooled_vs_integration")




# ---------------------------------------------------------------------
# Build figures.  `python make_figures.py`           -> builds all available
#                 `python make_figures.py fig5 fig2`  -> builds a subset
# Figures whose input CSV is missing are skipped with a message (e.g. fig6
# needs power_curve.csv + single_study_baselines.csv, which are not committed).
# ---------------------------------------------------------------------
_ALL = {"fig_diagnostics": fig_diagnostics, "fig2": fig2, "fig3": fig3,
        "fig4": fig4, "fig5": fig5, "fig6": fig6,
        "fig7_integration_quality": fig7_integration_quality}
 
if __name__ == "__main__":
    import sys
    want = sys.argv[1:] or list(_ALL)
    for key in want:
        fn = _ALL.get(key)
        if fn is None:
            print(f"skip: unknown figure '{key}' (choose from: {', '.join(_ALL)})")
            continue
        try:
            fn()
        except FileNotFoundError as e:
            print(f"skip {key}: missing input file ({getattr(e, 'filename', e)}).")
