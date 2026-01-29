from pathlib import Path
import subprocess
from termcolor import colored


def main():
    dataset_root = Path("./save_dataset_with_splits")
    for dataset in dataset_root.iterdir():
        dataset_name = dataset.stem
        # -------------------------------------------------------------
        # INFO: status printing
        print("=" * 60)
        print_line = "Running for dataset {}".format(
            colored(dataset_name, "red", attrs=["bold"])
        )
        print(f"{print_line:^70s}")
        print("=" * 60)
        # -------------------------------------------------------------
        dataset_file_path = dataset / f"{dataset_name}.pth"
        command = f"python3 Canopy.py --dataset_path {dataset_file_path}"
        subprocess.run(command.split())


if __name__ == "__main__":
    main()
