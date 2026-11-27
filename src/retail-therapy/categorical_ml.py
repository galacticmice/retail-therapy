"""Local hierarchical TF-IDF matching and compact downstream features."""
import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import hashlib
import json
from itertools import chain
from multiprocessing import get_context
import os
from pathlib import Path
import signal
import stat
import time
import tempfile
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
from threadpoolctl import threadpool_limits
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = Path(__file__).with_name("product_taxonomy.json")
DESCRIPTIONS_PATH = TAXONOMY_PATH.with_name("product_taxonomy_descriptions.json")
EVENTS_PATH = ROOT / "data/processed/events_dataset.parquet"
WORK_PATH = ROOT / "data/prep/tfidf_predictions"
PREDICTION_TYPE = pa.binary(6)
MAPPING_KEY = b"top3.category_mapping"
BRIDGE = {
    "All_Beauty": ["Beauty and personal care"],
    "Beauty_and_Personal_Care": ["Beauty and personal care"],
    "Amazon_Fashion": ["Clothing and accessories"],
    "Clothing_Shoes_and_Jewelry": ["Clothing and accessories"],
    "Appliances": ["Kitchen and appliances", "Home"],
    "Arts_Crafts_and_Sewing": ["Arts and crafts"],
    "Automotive": ["Automotive"], "Baby_Products": ["Baby"],
    "Cell_Phones_and_Accessories": ["Electronics"], "Electronics": ["Electronics"],
    "Gift_Cards": ["Celebrations and subscriptions"],
    "Subscription_Boxes": ["Celebrations and subscriptions"],
    "Grocery_and_Gourmet_Food": ["Food and drink"],
    "Handmade_Products": ["Arts and crafts", "Clothing and accessories", "Home", "Celebrations and subscriptions"],
    "Health_and_Household": ["Health and wellness", "Home", "Beauty and personal care"],
    "Health_and_Personal_Care": ["Health and wellness", "Beauty and personal care"],
    "Home_and_Kitchen": ["Home", "Kitchen and appliances", "Tools and hardware", "Celebrations and subscriptions"],
    "Industrial_and_Scientific": ["Industrial and scientific", "Tools and hardware"],
    "Musical_Instruments": ["Music equipment"],
    "Office_Products": ["Office and stationery", "Electronics"],
    "Patio_Lawn_and_Garden": ["Garden and outdoor living", "Tools and hardware"],
    "Pet_Supplies": ["Pets"],
    "Sports_and_Outdoors": ["Sports and outdoors", "Clothing and accessories", "Electronics"],
    "Tools_and_Home_Improvement": ["Tools and hardware", "Home", "Kitchen and appliances"],
    "Toys_and_Games": ["Toys and hobbies", "Celebrations and subscriptions"],
    "Video_Games": ["Media and digital products", "Electronics"],
}
for _category in ("Books", "Kindle_Store", "CDs_and_Vinyl", "Digital_Music", "Magazine_Subscriptions", "Movies_and_TV", "Software"):
    BRIDGE[_category] = ["Media and digital products"]
_classifier = None


def normalize(value):
    return " ".join(value.lower().split()) if isinstance(value, str) else ""


def load_taxonomy(path=TAXONOMY_PATH):
    taxonomy = json.loads(Path(path).read_text())
    if not isinstance(taxonomy, dict) or len(taxonomy) != 148:
        raise ValueError("Expected 148 category names and IDs")
    if any(type(v) is not int for v in taxonomy.values()) or list(taxonomy.values()) != list(range(148)):
        raise ValueError("Preserve the existing category order and IDs 0 through 147")
    if any(len(k.split(" / ")) != 2 or not all(k.split(" / ")) for k in taxonomy):
        raise ValueError("Expected parent / fine category names")
    return taxonomy


def build_classifier(taxonomy, descriptions, word_weight=.7, char_weight=.3, cache_size=0):
    weights = np.asarray([word_weight, char_weight], dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("TF-IDF weights must be finite, nonnegative, and not both zero")
    if not 0 <= cache_size <= 4096:
        raise ValueError("Cache size must be within [0, 4096]")
    weights /= weights.sum()
    parents = {}
    for name, category_id in taxonomy.items():
        parents.setdefault(name.split(" / ")[0], []).append(category_id)
    eligible = {normalize(parent): tuple(ids) for parent, ids in parents.items()}
    for category, groups in BRIDGE.items():
        eligible[normalize(category.replace("_", " "))] = tuple(sorted({i for g in groups for i in parents[g]}))
    # Unknown is a source label, not a taxonomy category. Use title-only matching.
    eligible["unknown"] = tuple(taxonomy.values())
    documents = [name.split(" / ")[1] + ". " + descriptions.get(name, "") for name in taxonomy]
    vectors = []
    for weight, options in zip(weights, [dict(analyzer="word", ngram_range=(1, 2)), dict(analyzer="char_wb", ngram_range=(3, 5))]):
        if weight:
            vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", dtype=np.float32, norm="l2", **options)
            references = vectorizer.fit_transform(documents)
            grouped = {ids: references[list(ids)].T for ids in set(eligible.values())}
            vectors.append((float(weight), vectorizer, grouped))
    return dict(vectors=vectors, eligible=eligible, cache=OrderedDict(), cache_size=cache_size)


def init_classifier(taxonomy, descriptions, word_weight, char_weight, cache_size, child=True):
    global _classifier, _native_threads
    if child:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    _native_threads = threadpool_limits(limits=1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    _classifier = build_classifier(taxonomy, descriptions, word_weight, char_weight, cache_size)


def pack_scores(scores, category_ids):
    scores, category_ids = np.asarray(scores), np.asarray(category_ids)
    if scores.ndim != 2 or category_ids.shape != (scores.shape[1],):
        raise ValueError("Expected a score matrix and one ID per score column")
    if category_ids.dtype.kind not in "iu" or (category_ids < 0).any() or (category_ids > 147).any() or len(set(category_ids)) != len(category_ids):
        raise ValueError("Score columns must have distinct category IDs in [0, 147]")
    if scores.dtype.kind not in "fiu" or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Matching scores must be finite and nonnegative")
    scores = scores.astype(np.float64, copy=False)
    order = np.argsort(category_ids)
    scores, category_ids = scores[:, order], category_ids[order]
    total = scores.sum(axis=1, keepdims=True)
    if not np.isfinite(total).all():
        raise ValueError("Matching score sum overflow")
    best = np.argsort(-scores, axis=1, kind="stable")[:, :3]
    top_scores = np.take_along_axis(scores, best, axis=1)
    selected = np.divide(top_scores, total, out=np.zeros_like(top_scores), where=total > 0)
    ids = category_ids[best]
    packed = np.zeros((len(scores), 6), dtype=np.uint8)
    packed[:, 0::2] = 255
    count = best.shape[1]
    packed[:, :count*2:2] = np.where(top_scores > 0, ids, 255)
    packed[:, 1:count*2:2] = np.rint(np.clip(selected, 0, 1) * 255).astype(np.uint8)
    return packed


def predictions_to_arrow(packed):
    packed = np.ascontiguousarray(packed)
    if packed.dtype != np.uint8 or packed.ndim != 2 or packed.shape[1] != 6:
        raise ValueError("Expected uint8 (rows, 6)")
    return pa.Array.from_buffers(PREDICTION_TYPE, len(packed), [None, pa.py_buffer(packed)])


def decode_top3(packed):
    if isinstance(packed, pa.ChunkedArray):
        packed = packed.combine_chunks()
    if isinstance(packed, pa.Array):
        if packed.type != PREDICTION_TYPE or packed.null_count:
            raise ValueError("Expected non-null fixed_size_binary[6]")
        packed.validate(full=True)
        packed = (np.frombuffer(packed.buffers()[1], dtype=np.uint8, count=len(packed)*6,
                               offset=packed.offset*6).reshape(-1, 6) if len(packed) else np.empty((0, 6), dtype=np.uint8))
    packed = np.asarray(packed)
    if packed.dtype != np.uint8 or packed.ndim != 2 or packed.shape[1] != 6:
        raise ValueError("Expected uint8 (rows, 6)")
    ids = packed[:, 0::2].copy()
    missing = ids == 255
    if (((ids > 147) & ~missing).any() or (packed[:, 1::2][missing] != 0).any()
        or (missing[:, :-1] & ~missing[:, 1:]).any()
        or (packed[:, 1:5:2] < packed[:, 3:6:2]).any()):
        raise ValueError("Malformed packed predictions")
    for i, j in ((0, 1), (0, 2), (1, 2)):
        if ((ids[:, i] == ids[:, j]) & ~missing[:, i]).any():
            raise ValueError("Duplicate prediction category")
    return ids, packed[:, 1::2].astype(np.float32) / 255


def classify_titles(titles, categories, classifier):
    if len(titles) != len(categories):
        raise ValueError("Title and category batch lengths differ")
    packed = np.zeros((len(titles), 6), dtype=np.uint8)
    packed[:, 0::2] = 255
    groups, counts = {}, dict(missing_title=0, zero_match=0, cache_hits=0)
    for index, (title, category) in enumerate(zip(titles, categories)):
        title = normalize(title)
        category = normalize(category.replace("_", " ") if isinstance(category, str) else category)
        eligible = classifier["eligible"].get(category)
        if not eligible:
            continue
        elif not title:
            counts["missing_title"] += 1
        else:
            groups.setdefault((category, eligible), {}).setdefault(title, []).append(index)
    for (category, eligible), titles_to_rows in groups.items():
        cache = classifier["cache"]
        uncached = list(titles_to_rows)
        if classifier["cache_size"]:
            uncached = []
            for title, rows in titles_to_rows.items():
                key = (category, title)
                if key in cache:
                    packed[rows] = cache[key]
                    cache.move_to_end(key)
                    counts["cache_hits"] += len(rows)
                    if cache[key][0] == 255:
                        counts["zero_match"] += len(rows)
                else:
                    uncached.append(title)
        if uncached:
            scores = np.zeros((len(uncached), len(eligible)), dtype=np.float32)
            for weight, vectorizer, references in classifier["vectors"]:
                scores += weight * (vectorizer.transform(uncached) @ references[eligible]).toarray()
            result = pack_scores(scores, np.asarray(eligible))
            row_groups = [titles_to_rows[title] for title in uncached]
            lengths = np.fromiter(map(len, row_groups), dtype=np.intp)
            rows = np.fromiter(chain.from_iterable(row_groups), dtype=np.intp)
            packed[rows] = np.repeat(result, lengths, axis=0)
            counts["zero_match"] += int(lengths[result[:, 0] == 255].sum())
            if classifier["cache_size"]:
                for title, prediction in zip(uncached, result):
                    if len(title) <= 1024:
                        cache[(category, title)] = prediction.copy()
                        while len(cache) > classifier["cache_size"]:
                            cache.popitem(last=False)
    return packed, counts


def classify_batch(item):
    index, titles, categories = item
    packed, counts = classify_titles(titles, categories, _classifier)
    return index, packed, counts


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def make_visible(path):
    # macOS retains hidden flags when temporary files are renamed or hard-linked.
    if hasattr(os, "chflags"):
        flags = path.stat().st_flags
        if flags & stat.UF_HIDDEN:
            os.chflags(path, flags & ~stat.UF_HIDDEN)


def atomic_json(path, value):
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def ordered_results(tasks, workers, initargs, pool=None):
    if workers == 1:
        init_classifier(*initargs, child=False)
        for task in tasks:
            yield classify_batch(task)
        return
    if pool is None:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                                 initializer=init_classifier, initargs=initargs) as pool:
            yield from ordered_results(tasks, workers, initargs, pool)
        return
    pending = OrderedDict()
    try:
        for task in tasks:
            pending[task[0]] = pool.submit(classify_batch, task)
            if len(pending) >= workers*2:
                yield pending.popitem(last=False)[1].result()
        while pending:
            yield pending.popitem(last=False)[1].result()
    finally:
        for future in pending.values():
            future.cancel()


def read_descriptions(path, taxonomy):
    descriptions = json.loads(Path(path).read_text()) if Path(path).exists() else {}
    if any(not isinstance(value, str) for value in descriptions.values()):
        raise ValueError("Descriptions must be strings")
    missing = set(taxonomy) - descriptions.keys()
    if missing:
        warnings.warn(f"{len(missing)} categories have no description; using category names")
    return {name: descriptions.get(name, "") for name in taxonomy}


def classify_file(events=EVENTS_PATH, output=None, work_dir=WORK_PATH, batch_size=10000,
                  workers="auto", word_weight=.7, char_weight=.3, cache_size=0,
                  limit=None, keep_checkpoints=False):
    events = Path(events).resolve()
    output = events if output is None else Path(output).absolute()
    if output.is_symlink() or (output.exists() and output.resolve() != events):
        raise FileExistsError("Output already exists or aliases another file")
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError("Batch size and pilot limit must be positive")
    taxonomy = load_taxonomy()
    descriptions = read_descriptions(DESCRIPTIONS_PATH, taxonomy)
    initargs = (taxonomy, descriptions, word_weight, char_weight, cache_size)
    # Validate parameters before opening checkpoints or launching children.
    build_classifier(*initargs)
    with pq.ParquetFile(events) as source:
        schema = source.schema_arrow
        rows = source.metadata.num_rows
    if not {"title", "category"} <= set(schema.names):
        raise ValueError("Expected title and category columns")
    if "top3_predictions" in schema.names:
        schema = schema.remove(schema.get_field_index("top3_predictions"))
    original_columns = schema.names
    total = rows if limit is None else min(limit, rows)
    if total < rows and output.resolve() == events:
        raise ValueError("Limited pilots must use a separate output")
    directory = Path(work_dir)
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(version="tfidf-relative-weight-v2-unknown-fallback", source=str(events), sha256=digest(events),
                  rows=total, batch_size=batch_size, taxonomy=taxonomy, descriptions=descriptions,
                  bridge=BRIDGE, word_weight=word_weight, char_weight=char_weight,
                  sklearn=sklearn.__version__, numpy=np.__version__, pyarrow=pa.__version__)
    manifest = directory / "source.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("Checkpoint configuration changed; use a new work directory")
    atomic_json(manifest, config)
    batches = (total + batch_size - 1) // batch_size

    def checkpoint(index):
        return directory / f"{index:08}.parquet"

    def valid(index):
        path = checkpoint(index)
        if not path.exists():
            return False
        try:
            table = pq.read_table(path)
            decode_top3(table["top3_predictions"])
            return (table.num_rows == min(batch_size, total-index*batch_size)
                    and (table.schema.metadata or {}).get(b"batch") == str(index).encode())
        except (ValueError, pa.ArrowException, KeyError):
            return False

    completed = {i for i in range(batches) if valid(i)}
    def tasks():
        with pq.ParquetFile(events) as source:
            for index, batch in enumerate(source.iter_batches(batch_size=batch_size, columns=["title", "category"], use_threads=False)):
                if index >= batches:
                    break
                if index in completed:
                    continue
                batch = batch.slice(0, min(batch_size, total-index*batch_size))
                yield index, batch["title"].to_pylist(), batch["category"].to_pylist()

    if str(workers) == "auto":
        # Measure steady throughput on spaced samples, not repeated first-batch
        # cache hits or the one-time spawn cost of a multi-hour job.
        sample_tasks = []
        with pq.ParquetFile(events) as source:
            groups = np.linspace(0, max(0, source.num_row_groups-1), 8, dtype=int)
            for index, group in enumerate(sorted(set(groups)) if rows else []):
                sample = next(source.iter_batches(batch_size=min(batch_size, 5000),
                    row_groups=[int(group)], columns=["title", "category"], use_threads=False), None)
                if sample is not None:
                    sample_tasks.append((index, sample["title"].to_pylist(), sample["category"].to_pylist()))
        # Disable LRU reuse in both benchmark arms to avoid warmup bias.
        benchmark_args = (*initargs[:-1], 0)
        timings = {}
        for count in (1, 4):
            context = ProcessPoolExecutor(max_workers=count, mp_context=get_context("spawn"),
                initializer=init_classifier, initargs=benchmark_args) if count > 1 else nullcontext(None)
            with context as pool:
                if count == 1:
                    init_classifier(*benchmark_args, child=False)
                    for task in sample_tasks:
                        classify_batch(task)
                else:
                    list(ordered_results(iter(sample_tasks), count, benchmark_args, pool))
                start = time.perf_counter()
                if count == 1:
                    for task in sample_tasks:
                        classify_batch(task)
                else:
                    for _ in ordered_results(iter(sample_tasks), count, benchmark_args, pool):
                        pass
                timings[count] = time.perf_counter()-start
        workers = 4 if sample_tasks and timings[4] < timings[1]*.9 else 1
        print(f"Auto workers: {workers}; steady-throughput seconds {timings}")
        del sample_tasks
    workers = int(workers)
    if not 1 <= workers <= 4:
        raise ValueError("Use 1 through 4 workers")
    with tqdm(total=batches, initial=len(completed), desc="Classifying", unit=" batches") as progress:
        for index, packed, counts in ordered_results(tasks(), workers, initargs):
            field_schema = pa.schema([pa.field("top3_predictions", PREDICTION_TYPE, nullable=False)],
                                     metadata={b"batch": str(index).encode(), b"counts": json.dumps(counts).encode()})
            path = checkpoint(index)
            temporary = path.with_suffix(".parquet.partial")
            pq.write_table(pa.Table.from_arrays([predictions_to_arrow(packed)], schema=field_schema), temporary, compression="zstd")
            temporary.replace(path)
            progress.update(1)
    metadata = {key: value for key, value in (schema.metadata or {}).items()
                if not key.startswith(b"top3.")}
    metadata.update({MAPPING_KEY: json.dumps(taxonomy).encode(), b"top3.format": b"tfidf-relative-weight-v1;missing=255", b"top3.config": json.dumps(config).encode()})
    schema = schema.append(pa.field("top3_predictions", PREDICTION_TYPE, nullable=False)).with_metadata(metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f"{output.name}.", suffix=".partial", dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    counts = dict(missing_title=0, zero_match=0, cache_hits=0)
    try:
        with pq.ParquetFile(events) as source, pq.ParquetWriter(temporary, schema, compression="zstd") as writer, tqdm(
            total=total, desc="Writing", unit=" events"
        ) as progress:
            for index, batch in enumerate(source.iter_batches(batch_size=batch_size, columns=original_columns, use_threads=False)):
                if index >= batches:
                    break
                batch = batch.slice(0, min(batch_size, total-index*batch_size))
                table = pq.read_table(checkpoint(index))
                predictions = table["top3_predictions"].combine_chunks()
                decode_top3(predictions)
                if len(predictions) != len(batch):
                    raise ValueError("Checkpoint row alignment changed")
                batch_counts = json.loads(table.schema.metadata[b"counts"])
                for key in counts:
                    counts[key] += batch_counts.get(key, 0)
                writer.write_table(pa.Table.from_arrays(
                    [*(batch[name] for name in original_columns), predictions], schema=schema
                ))
                progress.update(len(batch))
        print("Verifying output and source...")
        if pq.ParquetFile(temporary).metadata.num_rows != total or digest(events) != config["sha256"]:
            raise ValueError("Source or output row count changed; refusing replacement")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        if output.resolve() == events:
            temporary.replace(output)
        else:
            os.link(temporary, output)
            temporary.unlink()
        make_visible(output)
    finally:
        temporary.unlink(missing_ok=True)
    if not keep_checkpoints:
        for index in range(batches):
            checkpoint(index).unlink()
        manifest.unlink()
    print(f"Completed {total:,} events. Diagnostics: {counts}")
    return output


def iter_training_batches(path, batch_size=100000, columns=None):
    with pq.ParquetFile(path) as source:
        stored = json.loads((source.schema_arrow.metadata or {}).get(MAPPING_KEY, b"{}"))
        if list(stored.items()) != list(load_taxonomy().items()):
            raise ValueError("Prediction category mapping differs from training taxonomy")
        if columns is not None:
            columns = list(dict.fromkeys([*columns, "top3_predictions"]))
        for batch in source.iter_batches(batch_size=batch_size, columns=columns, use_threads=False):
            ids, weights = decode_top3(batch["top3_predictions"])
            yield batch, ids, weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=EVENTS_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path, default=WORK_PATH)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--workers", choices=["auto", "1", "2", "3", "4"], default="auto")
    parser.add_argument("--word-weight", type=float, default=.7)
    parser.add_argument("--char-weight", type=float, default=.3)
    parser.add_argument("--cache-size", type=int, default=0)
    parser.add_argument("--keep-checkpoints", action="store_true")
    run = parser.add_mutually_exclusive_group(required=True)
    run.add_argument("--limit", type=int)
    run.add_argument("--full", action="store_true")
    args = parser.parse_args()
    values = vars(args)
    values.pop("full")
    classify_file(**values)


if __name__ == "__main__":
    main()
