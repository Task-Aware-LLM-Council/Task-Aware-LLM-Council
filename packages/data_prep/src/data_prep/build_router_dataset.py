import random
from datasets import DatasetDict, Features, Sequence, Value
from common.split import split_dev_test
from common.hf_io import to_dataset, save_datasetdict_parquet, push_to_hub
from data_prep.load_musique import load_musique
from data_prep.load_quality import load_quality
from data_prep.load_fever import load_fever
from data_prep.load_hardmath import load_hardmath
from data_prep.load_humaneval import load_humaneval

random.seed(42)


def main():
    all_records = []
    loaders = [
        (load_musique, 40),
        (load_quality, 30),
        (load_fever, 40),
        (load_hardmath, 40),
        (load_humaneval, 40),
    ]

    for loader_fn, count in loaders:
        all_records.extend(loader_fn(count))
    print("Here")

    # Example: mark hallucination subset
    fever_recs = [r for r in all_records if r["source_dataset"] == "FEVER"]
    musique_recs = [r for r in all_records if r["source_dataset"] == "MuSiQue"]
    quality_recs = [r for r in all_records if r["source_dataset"] == "QuALITY"]

    halluc_candidates = fever_recs[:15] + musique_recs[:8] + quality_recs[:7]
    halluc_ids = {r["id"] for r in halluc_candidates}
    for r in all_records:
        r["hallucination_subset"] = r["id"] in halluc_ids

    dev_records, test_records = [], []
    for source in ["MuSiQue", "QuALITY", "FEVER", "HARDMATH", "HumanEvalPlus"]:
        source_recs = [r for r in all_records if r["source_dataset"] == source]
        dev, test = split_dev_test(source_recs, dev_frac=0.6)
        for r in dev:
            r["split"] = "dev"
        for r in test:
            r["split"] = "mini-test"
        dev_records.extend(dev)
        test_records.extend(test)

    ds = DatasetDict(
        {
            "full": to_dataset(dev_records + test_records),
            "dev": to_dataset(dev_records),
            "mini_test": to_dataset(test_records),
        }
    )

    save_datasetdict_parquet(ds, "data/router_dataset")


if __name__ == "__main__":
    main()
