from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="McAuley-Lab/Amazon-Reviews-2023",
    repo_type="dataset",
    filename="raw/review_categories/All_Beauty.jsonl",
    local_dir="../../data",
)

print("Saved: ", path)
