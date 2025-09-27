# hyparams.py
# ---------------- Core configuration ----------------
SEED = 1

# Model/output roots
MODELS = "./jit_models"
OUTPUT = "./outputs"
PROPOSED_CHANGES = "./plans"
PLOTS = "./plots"
MODEL_EVALUATION = "./model_evaluation"
EXPERIMENTS = "./experiments"
RESULTS = "./flip_rates"

# ---------------- JIT Dataset configuration ----------------
# Where your raw JIT CSV lives (total.csv with all projects)
JIT_DATASET_PATH = "./Dataset/apachejit_total.csv"

# Where your preprocess step writes per-project train/test:
#   <RELEASE_DATASET>/<project>@0/{train.csv,test.csv,mapping.csv}
RELEASE_DATASET = "./Dataset/jit_preprocessed"

# (Optional legacy; some helpers/tools may still reference this)
PROJECT_DATASET = "./Dataset/project_dataset"

# ---------------- Experiment switches ----------------
EXPLAINER_TYPES = [
    "LIME",
    "LIME-HPO",   # keep hyphen to match runner choices
]

MODEL_TYPES = [
    "RandomForest",
    "SVM",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]

# All possible JIT metric columns
JIT_FEATURES = ['la','ld','nf','nd','ns','ent','ndev','age','nuc','aexp','arexp','asexp']

# Project filtering / split
MIN_COMMITS = 100
MIN_BUGGY_COMMITS = 20
TEMPORAL_SPLIT_RATIO = 0.8
