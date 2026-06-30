import sys, numpy, pandas, sklearn, xgboost, shap, matplotlib,  itertools, argparse, pathlib
print(sys.version)
for m in (numpy, pandas, sklearn, xgboost, shap, matplotlib, argparse): print(m.__name__, m.__version__)