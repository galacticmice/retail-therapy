import csv
import gzip
import hashlib
import os
import shutil
import signal
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
from io import TextIOWrapper
from multiprocessing import get_context
from pathlib import Path
from queue import Empty

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

CATEGORIES_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/all_categories.txt"
BASE_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/benchmark/0core/rating_only"

PREPFOLDER_PATH = (Path(__file__).resolve().parent / "../../data/prep").resolve()
RAWFOLDER_PATH = (Path(__file__).resolve().parent / "../../data/raw").resolve()
PROCESSEDFOLDER_PATH = (
    Path(__file__).resolve().parent / "../../data/processed"
).resolve()

cohort_ids: set[str]  # suppress lsp warning
PROGRESS_INTERVAL = 1_000_000
_progress_queue = None
_stop_event = None


def init_progress(progress_queue, stop_event, initializer):
    """Share progress events with the parent, then initialize worker data."""
    global _progress_queue, _stop_event
    _progress_queue = progress_queue
    _stop_event = stop_event
    # The parent handles Ctrl-C; workers close their resources cooperatively.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if initializer is not None:
        initializer()


class ShutdownRequested(Exception):
    pass


def check_shutdown():
    if _stop_event is not None and _stop_event.is_set():
        raise ShutdownRequested()


def run_category(func, item):
    try:
        check_shutdown()
        return func(item)
    except ShutdownRequested:
        return None


def report_progress(category, rows=0, phase="reading"):
    if _progress_queue is not None:
        _progress_queue.put((os.getpid(), category, rows, phase))


def get_category_list():
    """get list of all categories + unknown"""
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
        return str(category)

    check_shutdown()
    report_progress(category)

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

        rows_read = 0
        for rows_read, row in enumerate(rows, start=1):
            if rows_read % 10_000 == 0:
                check_shutdown()
            user_id = row["user_id"]
            counts[user_id] = counts.get(user_id, 0) + 1
            if rows_read % PROGRESS_INTERVAL == 0:
                report_progress(category, rows_read)

    temporary = destination.with_suffix(".gz.partial")

    check_shutdown()
    report_progress(category, rows_read, "writing")
    # Publish only a complete count file; interrupted categories restart.
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["user_id", "events"])

        for index, (user_id, number) in enumerate(counts.items()):
            if index % 10_000 == 0:
                check_shutdown()
            writer.writerow([user_id, number])

    check_shutdown()
    temporary.replace(destination)
    return str(category)


def launch_parallel_func(directory, func, text, init=None, *, output_suffix=".csv.gz"):
    """Skip completed outputs and render all worker progress in the parent."""

    category_urls = get_category_list()
    directory.mkdir(parents=True, exist_ok=True)
    pending = [
        item
        for item in category_urls.items()
        if not (directory / f"{item[0]}{output_suffix}").exists()
    ]
    completed = len(category_urls) - len(pending)

    for category in category_urls:
        if (directory / f"{category}{output_suffix}").exists():
            tqdm.write(f"Finished: {category:<35} Already completed prior.")

    interrupted = False
    with ExitStack() as display:
        overall = display.enter_context(
            tqdm(
                total=len(category_urls),
                initial=completed,
                desc="Collecting",
                unit=" categories",
                position=0,
            )
        )
        if pending:
            worker_lines = [
                display.enter_context(
                    tqdm(
                        desc="Waiting",
                        total=None,
                        position=position,
                        bar_format="Processing: {desc} {n:,.0f} read",
                        leave=False,
                    )
                )
                for position in range(1, min(4, len(pending)) + 1)
            ]
            worker_slots = {}

            # Workers send small status messages; only this process draws text.
            context = get_context("spawn")
            progress_queue = context.Queue()
            stop_event = context.Event()
            try:
                with ProcessPoolExecutor(
                    max_workers=4,
                    mp_context=context,
                    initializer=init_progress,
                    initargs=(progress_queue, stop_event, init),
                ) as pool:
                    active_tasks = set()
                    try:
                        active_tasks = {
                            pool.submit(run_category, func, item) for item in pending
                        }
                        while active_tasks:
                            done, active_tasks = wait(
                                active_tasks, timeout=0.1, return_when=FIRST_COMPLETED
                            )
                            while True:
                                try:
                                    pid, category, rows, phase = (
                                        progress_queue.get_nowait()
                                    )
                                except Empty:
                                    break
                                if pid not in worker_slots:
                                    worker_slots[pid] = worker_lines[len(worker_slots)]
                                line = worker_slots[pid]
                                if phase == "writing":
                                    line.bar_format = " "
                                    line.clear()
                                    continue
                                line.bar_format = "Processing: {desc} {n:,.0f} read"
                                line.set_description_str(
                                    f"{category:<35}", refresh=False
                                )
                                line.n = rows
                                line.refresh()

                            for task in done:
                                result = task.result()
                                overall.update(1)
                                # Refresh immediately even when completions are close together.
                                overall.refresh()
                                tqdm.write(f"Finished: {result}")
                    except KeyboardInterrupt:
                        interrupted = True
                        stop_event.set()
                        for task in active_tasks:
                            task.cancel()
                        tqdm.write(
                            "Stopping workers; waiting for active reads to finish..."
                        )
                        pool.shutdown(wait=True, cancel_futures=True)
            finally:
                progress_queue.close()
                progress_queue.join_thread()

    if interrupted:
        print(
            "\nStopped. Completed files are preserved; unfinished categories restart next run."
        )
        raise SystemExit(130)

    print()
    print(text)
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
        launch_cleanup(PREPFOLDER_PATH / "counts")
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


def init_cohort():
    """loads cohort as a parallel worker initializer for efficiency"""

    global cohort_ids
    with gzip.open(
        Path(f"{PREPFOLDER_PATH}/cohort_users.csv.gz"), "rt", encoding="utf-8"
    ) as source:
        cohort_ids = {row["user_id"] for row in csv.DictReader(source)}


def parquet_category(item):
    """download raw data file, sampling only from selected cohort"""

    category, url = item

    # the labels we need
    EVENT_SCHEMA = pa.schema(
        [
            ("user_id", pa.string()),
            ("parent_asin", pa.string()),
            ("category", pa.string()),
            ("timestamp", pa.int64()),
        ]
    )

    BATCH_SIZE = 100000

    temporary = RAWFOLDER_PATH / f"{category}.parquet.partial"
    destination = RAWFOLDER_PATH / f"{category}.parquet"

    if destination.exists():
        return str(f"{category:<35} parquest previously imported")

    batch = []
    check_shutdown()
    report_progress(category)

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        response.raw.auto_close = False

        source = TextIOWrapper(response.raw, encoding="utf-8-sig", newline="")
        rows = csv.DictReader(source)

        with pq.ParquetWriter(temporary, EVENT_SCHEMA, compression="zstd") as writer:
            rows_read = 0
            for rows_read, row in enumerate(rows, start=1):
                if rows_read % 10_000 == 0:
                    check_shutdown()
                if rows_read % PROGRESS_INTERVAL == 0:
                    report_progress(category, rows_read)
                if row["user_id"] not in cohort_ids:
                    continue  # skip user_ids not selected as cohort

                event = {
                    "user_id": row["user_id"],
                    "parent_asin": row["parent_asin"],
                    "category": category,
                    "timestamp": int(row["timestamp"]),
                }
                batch.append(event)

                # batch write into file
                if len(batch) >= BATCH_SIZE:
                    table = pa.Table.from_pylist(batch, schema=EVENT_SCHEMA)
                    writer.write_table(table)
                    batch.clear()

            check_shutdown()
            report_progress(category, rows_read, "writing")
            if batch:
                table = pa.Table.from_pylist(batch, schema=EVENT_SCHEMA)
                writer.write_table(table)

    check_shutdown()
    temporary.replace(destination)
    return str(category)


def launch_parquet_combine():
    """combines raw categorical parquet files into one cleaned dataset"""

    files = sorted(RAWFOLDER_PATH.glob("*.parquet"))

    if len(files) != 34:
        launch_cleanup(RAWFOLDER_PATH)
        print(f"Expected 34 files, but only found {len(files)}!")
        input("\033[3mPress Enter to abort...\033[0m")
        return

    expected_rows = sum(pq.ParquetFile(file).metadata.num_rows for file in files)

    PROCESSEDFOLDER_PATH.mkdir(parents=True, exist_ok=True)
    destination = PROCESSEDFOLDER_PATH / "events_dataset.parquet"
    temporary = destination.with_suffix(".parquet.partial")
    (
        pl.scan_parquet(files)
        .sort(["user_id", "timestamp"])
        .sink_parquet(
            temporary, compression="zstd", engine="streaming", maintain_order=True
        )
    )

    actual_rows = pq.ParquetFile(temporary).metadata.num_rows

    if actual_rows != expected_rows:
        launch_cleanup(RAWFOLDER_PATH)
        print(
            "Inconsistent data detected, actual row count != expected row count. \nImport raw data again!"
        )
        print("Performing cleanup...")
        input("\033[3mPress Enter to abort...\033[0m")

    temporary.replace(destination)
    print()
    print("Merging...")
    print(f"Merge completed: logged {actual_rows:,} events.")
    input("\033[3mPress Enter to continue...\033[0m")


def launch_cleanup(dir):
    # """deletes all previous prep files other than the lastest one."""

    # eligible_users = "eligible_users.csv.gz"

    # if (Path(f"{PREPFOLDER_PATH}/counts") / eligible_users).exists():
    #     for file in Path(f"{PREPFOLDER_PATH}/counts").iterdir():
    #         if file.is_file() and file.name != eligible_users:
    #             file.unlink()

    for file in dir.glob("*.partial"):
        if file.is_file():
            file.unlink()

    # print()
    # input("\033[3mPress Enter to continue...\033[0m")


def launch_reset():
    confirm = input("WARNING: RESTARTS ALL DATA PREPARATION!\n\nARE YOU SURE? [Y/n] ")

    if confirm.lower() != "y":
        return

    shutil.rmtree(PREPFOLDER_PATH)
    shutil.rmtree(RAWFOLDER_PATH)

    print()
    print(f"✓ Deleted: {PREPFOLDER_PATH}")
    print(f"✓ Deleted: {RAWFOLDER_PATH}")

    input("\033[3mPress Enter to continue...\033[0m")


def check():
    files = sorted(RAWFOLDER_PATH.glob("*.parquet"))
    total_events = sum(pq.ParquetFile(file).metadata.num_rows for file in files)

    print(f"Categories: {len(files)} / 34")
    print(f"Events: {total_events:,}")
    print("Matches expected:", total_events == 75_603_822)

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
        [1] Start user_id counter (~20 mins, ~5gb)
        [2] Merge and scan eligible users (~6 mins, ~700mb)
        [3] Hash-select 15% of users to decrease space
        [4] Import raw cohort data (~30 mins, ~2gb)
        [5] Merge cohort data

        [9] Cleanup all incomplete files (safe)
        [0] Erase and reset all preparation files

        [x] Exit
            """)

        filepaths = [
            PREPFOLDER_PATH,
            PREPFOLDER_PATH / "counts",
            RAWFOLDER_PATH,
            PROCESSEDFOLDER_PATH,
        ]

        # cleanup
        for directory in filepaths:
            launch_cleanup(directory)

        cin = input("\t>> ").lower()

        match cin:
            case "1":
                launch_parallel_func(
                    Path(f"{PREPFOLDER_PATH}/counts"),
                    count_category,
                    "Checkpoint: all categories collected. You may merge users now.",
                )

            case "2":
                launch_merge_counters()

            case "3":
                launch_select_cohort()

            case "4":
                launch_parallel_func(
                    RAWFOLDER_PATH,
                    parquet_category,
                    "Checkpoint: raw data imported. You may merge users now.",
                    init_cohort,
                    output_suffix=".parquet",
                )

            case "5":
                launch_parquet_combine()
                print("Combining...")

            case "8":
                check()

            case "x":
                os.system("cls" if os.name == "nt" else "clear")
                break

            case "9":
                print("Cleaning up...")
                for directory in filepaths:
                    launch_cleanup(directory)
                print("Cleaned.")
                input("\033[3mPress Enter to continue...\033[0m")

            case "0":
                launch_reset()

            case _:
                print("Invalid input, try again.")
