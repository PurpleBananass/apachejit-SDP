import subprocess
from itertools import product

# model_types = ["LightGBM", "CatBoost"]  

# model_types = ["LogisticRegression"]  
model_types = ["SVM"]  
# model_types = ["RandomForest"]  
model_types = [ "XGBoost","RandomForest", "SVM", "CatBoost", "LightGBM"]
# explainer_types = ["LIME"]
# explainer_types = ["LIME-HPO"]
explainer_types = [ "LIME-HPO", "LIME"]
# explainer_types = ["CfExplainer"]
# explainer_types = ["PyExplainer"]
for model, explainer in product(model_types, explainer_types):
    print(f"\n{'='*50}")
    print(f"Running {explainer} on {model}")
    print(f"{'='*50}\n")
    
    subprocess.run([
        "python", "plan_closest.py",
        "--model_type", model,
        "--explainer_type", explainer,
        # "--max-workers", "8"
    ])