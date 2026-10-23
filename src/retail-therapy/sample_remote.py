import hashlib
import json
from urllib.request import urlopen

category = "Health_and_Personal_Care"

url = (
    "https://huggingface.co/datasets/"
    "McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    f"raw/review_categories/{category}.jsonl"
)

kept = 0

with (
    urlopen(url, timeout=60) as source,
    open(f"{category}_sample.jsonl", "w", encoding="utf-8") as output,
):
    for line in source:
        review = json.loads(line)
        user_id = review["user_id"]

        bucket = int(hashlib.sha256(user_id.encode()).hexdigest(), 16) % 1000

        if bucket != 0:
            continue

        output.write(
            json.dumps(
                {
                    "user_id": user_id,
                    "product_id": review["parent_asin"],
                    "timestamp": review["timestamp"],
                    "category": category,
                }
            )
            + "\n"
        )

        kept += 1

print("Events retained: ", kept)
