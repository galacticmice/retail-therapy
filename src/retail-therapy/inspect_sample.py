import json
from datetime import UTC, datetime

histories = {}

with open("../../data/sample/All_Beauty_sample.jsonl", encoding="utf-8") as source:
    for line in source:
        event = json.loads(line)
        user_id = event["user_id"]

        if user_id not in histories:
            histories[user_id] = []

        histories[user_id].append(event)

if histories:
    user_id = max(histories, key=lambda id: len(histories[id]))
    history = sorted(histories[user_id], key=lambda event: event["timestamp"])

    print("User:", user_id)
    print("Events", len(history))

    print("DATE\t  TIME\t\tPRODUCT_ID")
    for event in history:
        event_time = datetime.fromtimestamp(event["timestamp"] / 1000, tz=UTC)
        print(
            event_time.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds"),
            event["product_id"],
        )

else:
    print("Sample empty")
