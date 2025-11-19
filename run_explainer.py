import concurrent.futures
import warnings
import os
from argparse import ArgumentParser

from Explainer.LIME_HPO import LIME_HPO, LIME_Planner
from sklearn.exceptions import ConvergenceWarning
# from Explainer.TimeLIME import TimeLIME
from tqdm import tqdm

from data_utils import get_true_positives, read_dataset, get_output_dir, get_model

warnings.filterwarnings("ignore", category=ConvergenceWarning)

warnings.filterwarnings("ignore", category=UserWarning)


def process_test_idx(
    test_idx, true_positives, train_data, model, output_path, explainer_type
):
    feature_cols = [c for c in train_data.columns if c != "target"]
    test_instance = true_positives.loc[test_idx, feature_cols]
    output_file = output_path / f"{test_idx}.csv"

    if output_file.exists():
        return None

    print(f"[START] idx={test_idx} pid={os.getpid()}")

    if explainer_type == "LIME":
        LIME_Planner(
            X_train=train_data[feature_cols],
            test_instance=test_instance,
            training_labels=train_data[["target"]],
            model=model,
            path=output_file,
        )

    elif explainer_type == "LIME-HPO":
        LIME_HPO(
            X_train=train_data[feature_cols],
            test_instance=test_instance,
            training_labels=train_data[["target"]],
            model=model,
            path=output_file,
        )

    print(f"[END]   idx={test_idx} pid={os.getpid()}")
    return os.getpid()


def run_single_project(
    train_data, test_data, project_name, model_type, explainer_type, verbose=True
):
    output_path = get_output_dir(project_name, explainer_type, model_type)
    model = get_model(project_name, model_type)
    true_positives = get_true_positives(model, train_data, test_data)

    if len(true_positives) == 0:
        print("No true positives found, skipping...")
        return
    print(len(true_positives))

    if explainer_type in ["LIME", "LIME-HPO"]:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_test_idx,
                    test_idx,
                    true_positives,
                    train_data,
                    model,
                    output_path,
                    explainer_type,
                )
                for test_idx in true_positives.index
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"{project_name}",
                disable=not verbose,
            ):
                out = future.result()
                if out is not None:
                    tqdm.write(f"Process {out} finished")

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--model_type", type=str, default="RandomForest")
    argparser.add_argument("--explainer_type", type=str, default="LIME-HPO")
    argparser.add_argument("--project", type=str, default="all")
    args = argparser.parse_args()

    projects = read_dataset()

    if args.project == "all":
        project_list = list(sorted(projects.keys()))
    else:
        project_list = args.project.split(" ")

    for project in tqdm(project_list, desc="Project", leave=True):
        print(project)
        train, test = projects[project]
        run_single_project(train, test, project, args.model_type, args.explainer_type)
