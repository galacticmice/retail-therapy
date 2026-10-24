import csv
import gzip
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

CATEGORIES_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/all_categories.txt"
BASE_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/benchmark/0core/rating_only"

PREPFOLDER_PATH = "../../data/prep"


def get_category_list():
    # get list of all categories + unknown
    response = requests.get(CATEGORIES_URL, timeout=30)
    response.raise_for_status()
    categories = response.text.split()

    # get individual category's url
    category_urls = {category: f"{BASE_URL}/{category}.csv" for category in categories}
    return category_urls


def count_category(category_dict):
    """each CPU process parses a category and counts the number of times a user_id appears.
    then saves it in a (category name).csv.gz file.
    a completed file means the count was done, partial means it was incomplete."""

    category, url = category_dict
    destination = Path(f"{PREPFOLDER_PATH}/counts") / f"{category}.csv.gz"

    # if file exists means it was already scanned once
    if destination.exists():
        return str(f"{category:<35} Already completed prior.")

    # open category and count number of time user appears
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        response.encoding = "utf-8-sig"

        rows = csv.DictReader(
            line.decode("utf-8-sig")
            if isinstance(line, bytes)  # placeholder for linter error
            else line
            for line in response.iter_lines()
        )

        counts = {}

        for row in rows:
            user_id = row["user_id"]
            counts[user_id] = counts.get(user_id, 0) + 1

    temporary = destination.with_suffix(".gz.partial")

    # write to temporary file, checkpointing for interrupts
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["user_id", "events"])

        for user_id, number in counts.items():
            writer.writerow([user_id, number])

    temporary.replace(destination)
    return str(destination)


def launch_category_counter():
    category_urls = get_category_list()

    # create temporary folder for counting user events
    Path("../../data").mkdir(exist_ok=True)
    Path("../../data/prep").mkdir(exist_ok=True)
    Path(f"{PREPFOLDER_PATH}/counts").mkdir(exist_ok=True)

    # prep process pool for parallel counter
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = pool.map(count_category, category_urls.items())

        # printing progress
        for filename in tqdm(
            results,
            total=len(category_urls),
            desc="Progress",
        ):
            tqdm.write(f"Finished: {filename}")

    print()
    print("Checkpoint: all categories collected. You may merge users now.")
    input("\033[3mPress Enter to continue...\033[0m")


def launch_merge_counters():
    """takes all counters for all categories and tallies them up."""

    destination = Path(f"{PREPFOLDER_PATH}/eligible_users.csv.gz")
    temporary = Path(f"{PREPFOLDER_PATH}/eligible_users.csv.gz.partial")

    if destination.exists():
        print()
        print("Already counted users before - you can proceed to the next step!")
        input("\033[3mPress Enter to continue...\033[0m")
        return

    files = sorted(Path(f"{PREPFOLDER_PATH}/counts").glob("*.csv.gz"))
    if len(files) != 34:
        print()
        print(f"Expected 34 categories, but only found {len(files)} files.")
        print("Program must have been interrupted--run the counter again.")

        input("\033[3mPress Enter to continue...\033[0m")
        return

    print("It will be stuck at 100% for a few minutes--that is normal!")

    counts = {}

    # adds up all counters in all files
    for filename in tqdm(files, desc="Merge Progress"):
        with gzip.open(filename, "rt", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                user_id = row["user_id"]
                num_events = int(row["events"])

                counts[user_id] = counts.get(user_id, 0) + num_events

    # creates new file and writes the finalized count
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["user_id", "events"])
        writer.writerows(
            (user_id, num_events)
            for user_id, num_events in counts.items()
            if num_events >= 3
        )

    temporary.replace(destination)

    print()
    print("Total users: ", len(counts))
    print("Eligible users: ", sum(n >= 3 for n in counts.values()))

    print()
    input("\033[3mPress Enter to continue...\033[0m")


def launch_cleanup():
    """deletes all previous prep files other than the lastest one."""
    eligible_users = "eligible_users.csv.gz"

    if (Path(f"{PREPFOLDER_PATH}/counts") / eligible_users).exists():
        for file in Path(f"{PREPFOLDER_PATH}/counts").iterdir():
            if file.is_file() and file.name != eligible_users:
                file.unlink()

    for file in Path(f"{PREPFOLDER_PATH}/counts").glob("*.partial"):
        if file.is_file():
            file.unlink()

    print()
    input("\033[3mPress Enter to continue...\033[0m")


def launch_reset():
    confirm = input("WARNING: RESTARTS ALL DATA PREPARATION!\nARE YOU SURE? [y/N] ")

    if confirm.lower() != "y":
        return

    shutil.rmtree(PREPFOLDER_PATH)

    print(f"✓ Deleted: {PREPFOLDER_PATH}")
    print()
    input("\033[3mPress Enter to continue...\033[0m")


if __name__ == "__main__":
    # # get list of all categories + unknown
    # response = requests.get(CATEGORIES_URL, timeout=30)
    # response.raise_for_status()
    # categories = response.text.split()

    # # get individual category's url
    # category_urls = {category: f"{BASE_URL}/{category}.csv" for category in categories}

    # create temporary folder for counting user events
    # Path("counts").mkdir(exist_ok=True)

    # # for each category, map a user counter
    # with ProcessPoolExecutor(max_workers=4) as pool:
    #     results = pool.map(count_category, category_urls.items())

    #     # printing progress
    #     for filename in tqdm(
    #         results,
    #         total=len(category_urls),
    #         desc="Progress",
    #     ):
    #         tqdm.write(f"Finished: {filename}")
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("""
        [1] Start user_id counter (~20 mins, ~5GB space)
        [2] Merge and scan eligible users (~6 mins, ~700MB)

        [9] Cleanup unused data files
        [0] Erase and reset all data files

        [x] Exit
            """)
        cin = input("\t>> ").lower()

        match cin:
            case "1":
                launch_category_counter()

            case "2":
                launch_merge_counters()

            case "x":
                os.system("cls" if os.name == "nt" else "clear")
                break

            case "9":
                print("Cleaning up...")
                launch_cleanup()

            case "0":
                print("Deleting files...")
                launch_reset()

            case _:
                print("Invalid input, try again.")
