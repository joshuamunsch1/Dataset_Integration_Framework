# A framework for investigating the effectiveness of integrating heterogeneous spaceflight transcriptomic datasets


Analysis code and processed data for the manuscript
A framework for investigating the effectiveness of integrating heterogeneous spaceflight transcriptomic datasets (J. Munsch, G. C. M. Siontis)

The project benchmarks a modular integration framework — four batch-correction
strategies (per-study z-score, DESeq2+limma, RUVg, ComBat-ref) × four
label-prediction classifiers (SVM, ridge logistic regression, elastic net,
XGBoost), routed to GSEA or ORA against a pooled differential-expression
baseline and an independent random-effects (DerSimonian–Laird) meta-analysis,
on murine **liver** (six-batch validation benchmark) and **heart**
(three-dataset application) spaceflight transcriptomes from NASA OSDR/GeneLab.

---

## Repository layout

```
.
├── README.md
├── LICENSE                     # MIT
├── Environment
│    └── renv.lock
│    └── requirements.txt 
├── figure_data/                # the CSVs make_figures.py consumes 
├── make_figures.py             # regenerates all figures from figure_data/
├── Liver/                      # liver specific analyis files
│   ├── data/                   #   raw data, intermediate CSVs
│   └── pipeline/               #   merged_counts.rds, meta.rds, corrected matrices
├── Heart/                      # heart specific analyis files
│   ├── data/
│   └── pipeline/
├── 01_combat_ref, ...          # files applied to both datasets/framework tests


---

## Data

### Raw data
All raw gene-level count matrices are publicly available without restriction from
the NASA Open Science Data Repository (OSDR/GeneLab, <https://osdr.nasa.gov/>):

| Tissue | Accessions |
|--------|------------|
| Heart  | OSD-599, OSD-270, OSD-580 |
| Liver  | GLDS-47, GLDS-168 (RR-1 and RR-3 partitions), GLDS-242, GLDS-245, GLDS-379 |

Dataset DOIs are listed in Table 1 of the manuscript.

### Processed data — stored here
- `liver/data/`, `heart/data/` — per-study DESeq2 differential-expression tables
  and intermediate objects are not stored but can easily be reproduced from raw data
- `figure_data/` — the exact CSVs needed to regenerate every figure (these are
  the "values behind the graphs" required by the PLOS data policy).


---

## Software environment

### R
- R version: **4.6.0** .
- Package versions are pinned in `renv.lock`. Reproduce with:
  ```r
  install.packages("renv"); renv::restore()
  ```
- Key packages: `DESeq2`, `limma`, `edgeR`, `clusterProfiler`, `fgsea`,
  `RUVSeq`, `sva` (ComBat), `org.Mm.eg.db`, `dplyr`.

### Python
- Python version: **3.8.1**
- Package versions are pinned in `requirements.txt`. Reproduce with:
  ```bash
  pip install -r requirements.txt
  ```
- Key packages: `scikit-learn`, `xgboost`, `shap`, `numpy`, `pandas`,
  `matplotlib`.

A full `sessionInfo()` dump for the analysis machine is appended at the bottom
of this file.

---

## How to run

1. Clone the repository.
2. Restore the environments (`renv::restore()` and `pip install -r requirements.txt`).
3. Run the scripts the order indicated below. 

### Pipeline order

| Step | Script | Purpose |
|------|--------|---------|
| Normalise | `normalise_liver.R` / `normalise_heart.R` | gene and samples filtering + library-size normalisation |
| Merge | `liver_merge.R` / `heart_merge.R` | merge raw counts (liver / heart) |
| Correct | `01_combat_ref.R` | ComBat-ref correction  |
| Correct | `02_ruvg.R` | RUVg correction |
| Correct | `03_deseq2.R` | DESeq2 variance stabilisation + limma batch removal |
| Diagnose | `diagnostics.R` | silhouette / PCA batch-correction diagnostics |
| Model | `fit_liver.py` / `fit_heart.py` | classifiers (SVM/GLM/elastic-net/XGBoost) + SHAP |
| Targets | `10_dge_singlestudy_liver.R` / `10_dge_singlestudy_heart.R` | per-study DESeq2 recovery targets |
| Enrich | `11_gsea_liver.R` / `11_gsea_heart.R`, `11b_ora_liver.R` / `11b_ora_heart.R` | GSEA / ORA enrichment |
| Overlap | `13_go_overlap_methods_liver.R` / `13_go_overlap_methods_heart.R` | cross-arm term overlap |
| Stability | `14_power_permutation_loso.R` | within-study permutation + leave-one-study-out |
| Diagnostics | `17_integration_diagnostics.R` | D1/D2 integration diagnostics + meta referee |
| Control | `18_model_free_enrichment.R` | model-independent GSEA/ORA on pooled DE |
| Baseline | `19_pooled_vs_meta_referee.R` | pooled-DE baseline vs meta-analytic referee |
| Figures | `make_figures.py` | regenerate all figures from `figure_data/` |

### Figure → data map (`make_figures.py`)

| Figure(s) | Input CSV(s) in `figure_data/` |
|-----------|--------------------------------|
| Fig 2 (framework + recovery) | `merged_vs_single_summary_liver.csv`, `merged_vs_single_summary_heart.csv` |
| Fig 3 / Fig 4 (cardiac programmes, convergence) | `merged_vs_single_details_heart.csv` |
| Fig 5 (cross-arm overlap) | `between_arms_jaccard_liver.csv` |
| Fig 6 (power / permutation / LOSO) | `power_curve.csv`, `single_study_baselines.csv`, `loso.csv`, `permutation_null.csv` |
| Fig 7 + pooled-vs-integration | `d1_retention_summary.csv`, `d1_denoising_loso_meta.csv`, `d1_singleonly_reliability.csv`, `d2_merged_only_verdict.csv`, `d2_discovery_summary.csv` |
| Batch-correction diagnostics | `diagnostics_metrics.csv` |

## License

This repository is released under the MIT License (see `LICENSE`). 

---

## Citation

If you use this code or the processed data, please cite the manuscript and the
archived release:

- Munsch J, Siontis GCM. A framework for investigating the effectiveness of integrating heterogeneous spaceflight transcriptomic datasets. 
- Archived snapshot: Zenodo, DOI **[10.5281/zenodo.XXXXXXX]**.

---

## sessionInfo()

```
R version 4.6.0 (2026-04-24 ucrt)
Platform: x86_64-w64-mingw32/x64
Running under: Windows 10 x64 (build 19045)

Matrix products: default
  LAPACK version 3.12.1

locale:
[1] LC_COLLATE=German_Switzerland.utf8  LC_CTYPE=German_Switzerland.utf8    LC_MONETARY=German_Switzerland.utf8 LC_NUMERIC=C                       
[5] LC_TIME=German_Switzerland.utf8    

time zone: Europe/Zurich
tzcode source: internal

attached base packages:
[1] stats     graphics  grDevices utils     datasets  methods   base     

loaded via a namespace (and not attached):
 [1] Matrix_1.7-5                limma_3.68.4                gtable_0.3.6                dplyr_1.2.1                 compiler_4.6.0             
 [6] tidyselect_1.2.1            Rcpp_1.1.1-1.1              SummarizedExperiment_1.42.0 Biobase_2.72.0              GenomicRanges_1.64.0       
[11] parallel_4.6.0              IRanges_2.46.0              Seqinfo_1.2.0               scales_1.4.0                statmod_1.5.2              
[16] BiocParallel_1.46.0         lattice_0.22-9              ggplot2_4.0.3               R6_2.6.1                    XVector_0.52.0             
[21] S4Arrays_1.12.0             generics_0.1.4              BiocGenerics_0.58.1         tibble_3.3.1                DelayedArray_0.38.2        
[26] MatrixGenerics_1.24.0       pillar_1.11.1               RColorBrewer_1.1-3          rlang_1.2.0                 S7_0.2.2                   
[31] SparseArray_1.12.2          cli_3.6.6                   magrittr_2.0.5              locfit_1.5-9.12             grid_4.6.0                 
[36] rstudioapi_0.18.0           edgeR_4.10.1                lifecycle_1.0.5             DESeq2_1.52.0               S4Vectors_0.50.1           
[41] vctrs_0.7.3                 glue_1.8.1                  farver_2.1.2                codetools_0.2-20            abind_1.4-8                
[46] stats4_4.6.0                pkgconfig_2.0.3             matrixStats_1.5.0           tools_4.6.0
```
