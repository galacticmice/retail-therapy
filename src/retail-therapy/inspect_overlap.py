import json

user_categories = {}

files = [
    "../../data/sample/All_Beauty_sample.jsonl",
    "Health_and_Personal_Care_sample.jsonl",
]


for filename in files:
    with open(filename, encoding="utf-8") as source:
        for line in source:
            event = json.loads(line)
            user_id = event["user_id"]

            if user_id not in user_categories:
                user_categories[user_id] = set()

            user_categories[user_id].add(event["category"])

overlap = sum(len(categories) > 1 for categories in user_categories.values())


print("Unique users: ", len(user_categories))
print("Users in both categories: ", overlap)

example_user = next(
    user_id for user_id, categories in user_categories.items() if len(categories) > 1
)

history = []

for filename in files:
    with open(filename, encoding="utf-8") as source:
        for line in source:
            event = json.loads(line)

            if event["user_id"] == example_user:
                history.append(event)

history.sort(key=lambda event: event["timestamp"])

for event in history:
    print(
        event["timestamp"],
        event["category"],
        event["product_id"],
    )
