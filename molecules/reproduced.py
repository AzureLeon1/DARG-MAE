import os

current_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(current_dir, "checkpoints")

device = 3


def run_finetuning(datasets):
    for dataset in datasets:
        for seed in range(10):
            input_model_file = os.path.join(
                base_path, f"save_{sub_weight[dataset]}_{uniformity_dim[dataset]}_gin_100.pth")
            finetune_command = f"sh scripts/run_{dataset}.sh {dataset} {input_model_file} {seed} {device}"
            print(
                f"Running finetuning for {dataset} with seed {seed} and sub_weight {sub_weight[dataset]}...")
            os.system(finetune_command)


# List of datasets
datasets = ["bbbp", "tox21", "toxcast",
            "sider", "clintox", "muv", "hiv", "bace"]
uniformity_dim = {"bbbp": 8, "tox21": 4, "toxcast": 32,
                  "sider": 64, "clintox": 128, "muv": 8, "hiv": 4, "bace": 16}
sub_weight = {"bbbp": 0.8, "tox21": 0.4, "toxcast": 0.4,
              "sider": 0.2, "clintox": 0.3, "muv": 0.4, "hiv": 0.8, "bace": 0.2}
run_finetuning(datasets)
