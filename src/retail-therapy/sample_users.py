import hashlib
import json

user_counts = {}

with (
    open(
        "../../data/raw/review_categories/All_Beauty.jsonl", encoding="utf-8"
    ) as source,
    open("../../data/sample/All_Beauty_sample.jsonl", "w", encoding="utf-8") as output,
):
    for line in source:
        review = json.loads(line)
        user_id = review["user_id"]

        selection_bucket = int(hashlib.sha256(user_id.encode()).hexdigest(), 16) % 1000

        if selection_bucket != 0:
            continue

        output.write(
            json.dumps(
                {
                    "user_id": user_id,
                    "product_id": review["parent_asin"],
                    "timestamp": review["timestamp"],
                    "category": "All_Brauty",
                }
            )
            + "\n"
        )

        user_counts[user_id] = user_counts.get(user_id, 0) + 1


print("Users retained: ", len(user_counts))
print("Events retained: ", sum(user_counts.values()))
print("Users with 2+ events: ", sum(n >= 2 for n in user_counts.values()))
