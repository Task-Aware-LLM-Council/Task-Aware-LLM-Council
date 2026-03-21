from pathlib import Path
from datasets import Dataset, DatasetDict


def to_dataset(records: list, features=None) -> Dataset:
    return Dataset.from_list(records, features=features)


def save_parquet(dataset: Dataset, path: str):
    dataset.to_parquet(path)


def save_datasetdict_parquet(ds: DatasetDict, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split_name, split_ds in ds.items():
        split_ds.to_parquet(str(out / f"{split_name}.parquet"))


def push_to_hub(dataset_dict: DatasetDict, repo_id: str):
    dataset_dict.push_to_hub(repo_id)
