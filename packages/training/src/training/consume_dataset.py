from datasets import load_dataset

from common.schema import RouterExample

# latest
ds = load_dataset("task-aware-llm-council/router-dataset")

print(ds)

validation_rows = [RouterExample(**row) for row in ds["validation"]]
test_rows = [RouterExample(**row) for row in ds["test"]]

for item in validation_rows[-1:-3:-1]:
    print(item.question)
    print(item.skill_tags)
    print(item.source_dataset)
