import csv
import gzip
import hashlib
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

CATEGORIES_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/all_categories.txt"
BASE_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/benchmark/0core/rating_only"

PREPFOLDER_PATH = (Path(__file__).resolve().parent / "../../data/prep").resolve()


def get_category_list():
    # get list of all categories + unknown
    response = requests.get(CATEGORIES_URL, timeout=30)
    response.raise_for_status()
    categories = response.text.split()

    # get individual category's url
    category_urls = {category: f"{BASE_URL}/{category}.csv" for category in categories}
    return category_urls


def count_category(category_dict):
    """worker function for processes to run to count users in categories"""

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
    """each CPU process parses a category and counts the number of times a user_id appears.
    then saves it in a (category name).csv.gz file.
    a completed file means the count was done, partial means it was incomplete."""

    category_urls = get_category_list()

    # create temporary folder for counting user events
    Path(f"{PREPFOLDER_PATH}/counts").mkdir(parents=True, exist_ok=True)

    # prep process pool for parallel counter

    with ProcessPoolExecutor(
        max_workers=max(4, (os.process_cpu_count() or 1) - 2)
    ) as pool:
        active_tasks = [
            pool.submit(count_category, item) for item in category_urls.items()
        ]

        for task in tqdm(
            as_completed(active_tasks),
            total=len(active_tasks),
            desc="Counting Users",
            unit=" categories",
        ):
            filename = task.result()
            tqdm.write(f"Finished: {filename}")

    print()
    print("Checkpoint: all categories collected. You may merge users now.")
    input("\033[3mPress Enter to continue...\033[0m")


def launch_merge_counters():
    """takes all counters for all categories and tallies them up."""

    destination = Path(f"{PREPFOLDER_PATH}/eligible_users.csv.gz")
    temporary = Path(f"{PREPFOLDER_PATH}/eligible_users.csv.gz.partial")

    try:
        print()
        desired_events = int(input("Minimum events to qualify(Enter=default 5)? "))
        if desired_events < 1:
            desired_events = 5
    except ValueError:
        desired_events = 5

    # if destination.exists():
    #     print()
    #     print("Already counted users before - you can proceed to the next step!")
    #     input("\033[3mPress Enter to continue...\033[0m")
    #     return

    files = sorted(Path(f"{PREPFOLDER_PATH}/counts").glob("*.csv.gz"))
    if len(files) != 34:
        print()
        print(f"Expected 34 categories, but only found {len(files)} files.")
        print("Program must have been interrupted--run the counter again.")

        input("\033[3mPress Enter to continue...\033[0m")
        return

    print()
    print("It will be stuck at 100% for a few minutes--that is normal!")

    counts = {}

    # adds up all counters in all files
    for filename in tqdm(files, desc="Merge Progress", unit=" files"):
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
            if num_events >= desired_events
        )

    temporary.replace(destination)

    print()
    print("Total users: ", len(counts))
    print(
        f"Eligible users(>={desired_events} events): ",
        sum(n >= desired_events for n in counts.values()),
    )

    print()
    input("\033[3mPress Enter to continue...\033[0m")


def launch_select_cohort():
    """selects a % of the eligible users to track for training.
    default is 15% -- reduces file size."""

    selected_users = 0
    selected_events = 0

    try:
        sample_percent = int(input("% of users to sample from(Enter=default 15%): "))
        if sample_percent < 1:
            sample_percent = 15
    except ValueError:
        sample_percent = 15

    temporary = Path(f"{PREPFOLDER_PATH}/cohort_users.csv.gz.partial")
    destination = Path(f"{PREPFOLDER_PATH}/cohort_users.csv.gz")

    with (
        gzip.open(
            f"{PREPFOLDER_PATH}/eligible_users.csv.gz", "rt", encoding="utf-8"
        ) as source,
        gzip.open(temporary, "wt", encoding="utf-8", newline="") as output,
    ):
        rows = tqdm(
            csv.DictReader(source),
            desc="Selection Progress",
            unit=" users",
        )
        selected = (
            row
            for row in rows
            if int(
                hashlib.sha256(row["user_id"].encode()).hexdigest(),
                16,
            )
            % 100
            < sample_percent
        )

        writer = csv.writer(output)
        writer.writerow(["user_id", "events"])

        for row in selected:
            writer.writerow([row["user_id"], row["events"]])
            selected_users += 1
            selected_events += int(row["events"])

    if selected_users < 100_000:
        print("Cohort has fewer than 100,000 users. Try increasing sample %")

    temporary.replace(destination)

    print()
    print("File created: ../../data/prep/cohort_users.csv.gz")
    print("Selected users: ", selected_users)
    print("Their total events: ", selected_events)
    input("\033[3mPress Enter to continue...\033[0m")


def launch_cleanup():
    # """deletes all previous prep files other than the lastest one."""

    # eligible_users = "eligible_users.csv.gz"

    # if (Path(f"{PREPFOLDER_PATH}/counts") / eligible_users).exists():
    #     for file in Path(f"{PREPFOLDER_PATH}/counts").iterdir():
    #         if file.is_file() and file.name != eligible_users:
    #             file.unlink()

    for file in Path(f"{PREPFOLDER_PATH}/counts").glob("*.partial"):
        if file.is_file():
            file.unlink()

    # print()
    # input("\033[3mPress Enter to continue...\033[0m")


def launch_reset():
    confirm = input("WARNING: RESTARTS ALL DATA PREPARATION!\n\nARE YOU SURE? [Y/n] ")

    if confirm.lower() != "y":
        return

    shutil.rmtree(PREPFOLDER_PATH)

    print()
    print(f"✓ Deleted: {PREPFOLDER_PATH}")
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
        [3] Hash-select 15% of users to decrease space

        [0] Erase and reset all preparation files

        [x] Exit
            """)

        launch_cleanup()
        cin = input("\t>> ").lower()

        match cin:
            case "1":
                launch_category_counter()

            case "2":
                launch_merge_counters()

            case "3":
                launch_select_cohort()

            case "x":
                os.system("cls" if os.name == "nt" else "clear")
                break

            # case "9":
            #     print("Cleaning up...")
            #     launch_cleanup()

            case "0":
                launch_reset()

            case _:
                print("Invalid input, try again.")
