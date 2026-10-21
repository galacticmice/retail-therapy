from datasets import load_dataset

data_path = "../../data/raw/review_categories/All_Beauty.jsonl"

reviews = load_dataset(
    "json", data_files={"full": data_path}, split="full", streaming=True
)

print(next(iter(reviews)))
