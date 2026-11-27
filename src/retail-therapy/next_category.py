import argparse
import json
import os
import signal
import time
from collections import deque
from contextlib import contextmanager
from multiprocessing import get_context
from pathlib import Path
from queue import Empty

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from categorical_ml import EVENTS_PATH, MAPPING_KEY, decode_top3, load_taxonomy
from scipy import sparse
from threadpoolctl import threadpool_limits
from tqdm import tqdm

DAY_MS = 86_400_000
TIMESCALES = np.array([1.0, 7.0, 30.0, 180.0])
CATEGORY_COUNT = 148
FEATURE_COUNT = 740
MODEL_PATH = Path(__file__).resolve().parents[2] / "data/models/next_category.npz"


def validate_history(timestamps, category_ids, category_weights):
    timestamps = np.asarray(timestamps)
    ids, weights = np.asarray(category_ids), np.asarray(category_weights)
    if timestamps.shape == (0,) and ids.size == 0 and weights.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0, 3), dtype=np.float32),
        )
    if timestamps.ndim != 1 or timestamps.dtype.kind not in "iu":
        raise ValueError("Timestamps must be integer Unix milliseconds")
    if ids.shape != (len(timestamps), 3) or weights.shape != ids.shape:
        raise ValueError("Expected category IDs and weights shaped (events, 3)")
    if ids.dtype.kind not in "iu" or ((ids < 0) | ((ids > 147) & (ids != 255))).any():
        raise ValueError("Category IDs must be 0 through 147 or sentinel 255")
    if (
        weights.dtype.kind not in "fiu"
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or (weights > 1).any()
    ):
        raise ValueError("Category weights must be finite and within [0, 1]")
    if (weights[ids == 255] != 0).any() or (
        weights.sum(axis=1) > 1 + 3 / 510 + 1e-6
    ).any():
        raise ValueError("Invalid sentinel weights or retained mass")
    for i, j in ((0, 1), (0, 2), (1, 2)):
        if ((ids[:, i] == ids[:, j]) & (ids[:, i] != 255)).any():
            raise ValueError("Duplicate categories within an event")
    if (timestamps[1:] < timestamps[:-1]).any():
        raise ValueError("Histories must be chronological; ties retain supplied order")
    return timestamps, ids, weights


def new_state():
    return dict(
        history=np.zeros((4, CATEGORY_COUNT), dtype=np.float32),
        totals=np.zeros(CATEGORY_COUNT, dtype=np.float32),
        timestamp=None,
    )


def update_state(state, timestamp, ids, weights):
    timestamp = int(timestamp)
    if state["timestamp"] is not None:
        elapsed = timestamp - state["timestamp"]
        if elapsed < 0:
            raise ValueError("History timestamps decreased")
        state["history"] *= np.exp(-(elapsed / DAY_MS) / TIMESCALES)[:, None]
    valid = ids != 255
    state["history"][:, ids[valid]] += weights[valid]
    state["totals"][ids[valid]] += weights[valid]
    state["timestamp"] = timestamp
    return state


def history_features(state):
    total = float(state["totals"].sum())
    frequencies = state["totals"] / total if total else state["totals"]
    return np.concatenate((state["history"].ravel(), frequencies)).astype(
        np.float32, copy=False
    )


def new_model():
    return dict(
        weights=np.zeros((FEATURE_COUNT, CATEGORY_COUNT), dtype=np.float32),
        bias=np.zeros(CATEGORY_COUNT, dtype=np.float32),
        prior=np.zeros(CATEGORY_COUNT, dtype=np.float64),
        metadata=dict(
            version=1,
            taxonomy=load_taxonomy(),
            timescales=TIMESCALES.tolist(),
            timestamp_unit="unix_ms",
            step=0,
            epoch=0,
        ),
    )


def log_probabilities(model, features):
    # dense single-request inference
    product = (
        features @ model["weights"]
        if sparse.issparse(features)
        else np.einsum("ij,jk->ik", features, model["weights"])
    )
    logits = np.asarray(product + model["bias"], dtype=np.float64)
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite model logits")
    logits -= logits.max(axis=1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def predict_next(model, timestamps, category_ids, category_weights):
    """Return 148 probabilities after the last event; never mutate model or inputs."""
    timestamps, ids, weights = validate_history(
        timestamps, category_ids, category_weights
    )
    state = new_state()
    for timestamp, categories, values in zip(timestamps, ids, weights):
        update_state(state, timestamp, categories, values)
    return np.exp(log_probabilities(model, history_features(state)[None, :])[0]).astype(
        np.float32
    )


def loss_gradient(model, features, ids, weights, l2=1e-4):
    logs = log_probabilities(model, features)
    mass = weights.sum(axis=1)
    valid = ids != 255
    rows = np.broadcast_to(np.arange(len(ids))[:, None], ids.shape)[valid]
    labels, values = ids[valid], weights[valid]
    loss = -float(np.sum(values * logs[rows, labels])) / len(ids)
    residual = np.exp(logs) * mass[:, None]
    np.add.at(residual, (rows, labels), -values)
    residual = (residual / len(ids)).astype(np.float32)
    gradient = np.asarray(features.T @ residual) + l2 * model["weights"]
    loss += 0.5 * l2 * float(np.square(model["weights"], dtype=np.float64).sum())
    return loss, gradient, residual.sum(axis=0)


def train_batch(model, features, ids, weights, learning_rate=0.01, l2=1e-4):
    loss, gradient, bias_gradient = loss_gradient(model, features, ids, weights, l2)
    step = model["metadata"]["step"]
    rate = learning_rate / np.sqrt(1 + step / 1000)
    model["weights"] -= rate * gradient
    model["bias"] -= rate * bias_gradient
    model["metadata"]["step"] = step + 1
    return loss


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as output:
        np.savez_compressed(
            output,
            weights=model["weights"],
            bias=model["bias"],
            prior=model["prior"],
            metadata=json.dumps(model["metadata"]),
        )
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def load_model(path):
    with np.load(path, allow_pickle=False) as saved:
        model = {key: saved[key].copy() for key in ("weights", "bias", "prior")}
        model["metadata"] = json.loads(str(saved["metadata"]))
    metadata = model["metadata"]
    if (
        metadata.get("version") != 1
        or metadata.get("timescales") != TIMESCALES.tolist()
        or metadata.get("timestamp_unit") != "unix_ms"
        or list(metadata.get("taxonomy", {}).items()) != list(load_taxonomy().items())
    ):
        raise ValueError("Model mapping or feature definition is incompatible")
    for name, shape in (("weights", (740, 148)), ("bias", (148,)), ("prior", (148,))):
        if model[name].shape != shape or not np.isfinite(model[name]).all():
            raise ValueError("Malformed model parameters")
        model[name].flags.writeable = False
    return model


def inspect_source(path):
    with pq.ParquetFile(path) as source:
        schema = source.schema_arrow
        if not {"user_id", "timestamp", "top3_predictions"} <= set(schema.names):
            raise ValueError("Expected user_id, timestamp, and top3_predictions")
        if (
            schema.field("top3_predictions").type != pa.binary(6)
            or schema.field("timestamp").type != pa.int64()
        ):
            raise ValueError(
                "Expected fixed_size_binary[6] predictions and int64 millisecond timestamps"
            )
        mapping = json.loads((schema.metadata or {}).get(MAPPING_KEY, b"{}"))
        if list(mapping.items()) != list(load_taxonomy().items()):
            raise ValueError("Source taxonomy differs from predictor taxonomy")
        return source.metadata.num_rows


def read_batches(path, start, stop, columns):
    with pq.ParquetFile(path) as source:
        offsets = np.cumsum(
            [0]
            + [
                source.metadata.row_group(i).num_rows
                for i in range(source.num_row_groups)
            ]
        )
        groups = [
            i
            for i in range(source.num_row_groups)
            if offsets[i] < stop and offsets[i + 1] > start
        ]
        if not groups:
            return
        position = int(offsets[groups[0]])
        for batch in source.iter_batches(
            batch_size=65536, columns=columns, row_groups=groups, use_threads=False
        ):
            left, right = max(start - position, 0), min(stop - position, len(batch))
            if right > left:
                yield batch.slice(left, right - left)
            position += len(batch)
            if position >= stop:
                break


def user_ranges(path, rows, count):
    boundaries = [0]
    for proposed in [rows * i // count for i in range(1, count)]:
        first = None
        position = proposed
        for batch in read_batches(path, proposed, rows, ["user_id"]):
            users = batch["user_id"].to_numpy(zero_copy_only=False)
            if first is None:
                first = users[0]
            different = np.flatnonzero(users != first)
            if len(different):
                boundaries.append(position + int(different[0]))
                break
            position += len(batch)
    boundaries = sorted(set([*boundaries, rows]))
    return list(zip(boundaries[:-1], boundaries[1:]))


def time_cutoffs(path, rows):
    histogram = {}
    for batch in read_batches(path, 0, rows, ["timestamp"]):
        days = batch["timestamp"].to_numpy(zero_copy_only=False) // DAY_MS
        values, counts = np.unique(days, return_counts=True)
        for day, count in zip(values, counts):
            histogram[int(day)] = histogram.get(int(day), 0) + int(count)
    if not histogram:
        raise ValueError("Cannot train on an empty dataset")
    days = sorted(histogram)
    cumulative = np.cumsum([histogram[d] for d in days])
    cutoffs = [
        int(
            (days[min(np.searchsorted(cumulative, rows * fraction), len(days) - 1)] + 1)
            * DAY_MS
        )
        for fraction in (0.8, 0.9)
    ]
    if cutoffs[0] >= cutoffs[1]:
        raise ValueError(
            "Too few distinct days for automatic splits; provide explicit cutoffs"
        )
    return cutoffs


def feature_batches(path, start, stop, lower, upper, batch_size=1024, diagnostics=None):
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics.update(first_events=0, zero_mass_targets=0, examples=0)
    state, previous_user, previous_time = None, None, None
    features, labels, targets = [], [], []
    for batch in read_batches(
        path, start, stop, ["user_id", "timestamp", "top3_predictions"]
    ):
        ids, weights = decode_top3(batch["top3_predictions"])
        masses = weights.sum(axis=1)
        if (masses > 1 + 3 / 510 + 1e-6).any():
            raise ValueError("Invalid retained category mass")
        timestamps = batch["timestamp"].to_numpy(zero_copy_only=False)
        users = batch["user_id"].to_pylist()
        for user, timestamp, categories, values, mass in zip(
            users, timestamps, ids, weights, masses
        ):
            if not isinstance(user, str) or not isinstance(
                timestamp, (int, np.integer)
            ):
                raise ValueError(
                    "Expected string users and integer millisecond timestamps"
                )
            timestamp = int(timestamp)
            if previous_user is not None and user < previous_user:
                raise ValueError("Source must be grouped by increasing user_id")
            same_user = user == previous_user
            if same_user and timestamp < previous_time:
                raise ValueError("Source histories must be chronological")
            if lower <= timestamp < upper:
                key = (
                    "first_events"
                    if not same_user
                    else "examples"
                    if mass > 0
                    else "zero_mass_targets"
                )
                diagnostics[key] += 1
            if not same_user:
                state = new_state()
            elif lower <= timestamp < upper and mass > 0:
                # predict the target BEFORE applying its timestamp decay or weights
                features.append(history_features(state))
                labels.append(categories)
                targets.append(values)
            update_state(state, timestamp, categories, values)
            previous_user, previous_time = user, timestamp
            if len(features) == batch_size:
                yield (
                    sparse.csr_matrix(np.asarray(features)),
                    np.asarray(labels),
                    np.asarray(targets),
                )
                features, labels, targets = [], [], []
    if features:
        yield (
            sparse.csr_matrix(np.asarray(features)),
            np.asarray(labels),
            np.asarray(targets),
        )


def feature_worker(queue, arguments):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    pa.set_cpu_count(1)
    diagnostics = {}
    with threadpool_limits(limits=1):
        try:
            for batch in feature_batches(*arguments, diagnostics=diagnostics):
                queue.put(("batch", batch))
        except Exception as error:
            queue.put(("error", (type(error).__name__, str(error))))
        finally:
            queue.put(("done", diagnostics))


@contextmanager
def prepared_batches(
    path, rows, lower, upper, batch_size=1024, workers=0, diagnostics=None
):
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics.update(first_events=0, zero_mass_targets=0, examples=0)
    if not 0 <= workers <= 3:
        raise ValueError(
            "Use 0 through 3 feature workers, reserving a core for training"
        )
    if workers == 0:
        with threadpool_limits(limits=4):
            yield feature_batches(path, 0, rows, lower, upper, batch_size, diagnostics)
        return
    context = get_context("spawn")
    queues, processes = [], []
    try:
        for start, stop in user_ranges(path, rows, workers):
            queue = context.Queue(maxsize=2)
            process = context.Process(
                target=feature_worker,
                args=(queue, (str(path), start, stop, lower, upper, batch_size)),
            )
            queue.cancel_join_thread()
            process.start()
            queues.append(queue)
            processes.append(process)

        def consume():
            active = deque(range(len(processes)))
            while active:
                index = active.popleft()
                try:
                    kind, value = queues[index].get(timeout=1)
                except Empty:
                    if not processes[index].is_alive():
                        raise RuntimeError(
                            f"Feature worker exited: {processes[index].exitcode}"
                        )
                    active.appendleft(index)
                    continue
                if kind == "error":
                    raise RuntimeError(f"Feature worker {value[0]}: {value[1]}")
                if kind == "done":
                    for key, count in value.items():
                        diagnostics[key] += count
                if kind == "batch":
                    yield value
                    active.append(index)

        with threadpool_limits(limits=4 - workers):
            yield consume()
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()
        for queue in queues:
            queue.close()


def choose_workers(path, rows, lower, upper, batch_size):
    timings = {}
    # benchmark complete feature-generation + gradient steps, including IPC
    for workers in range(4):
        model = new_model()
        start = time.perf_counter()
        with prepared_batches(
            path, min(rows, 500000), lower, upper, batch_size, workers
        ) as batches:
            for features, ids, weights in batches:
                train_batch(model, features, ids, weights)
        timings[workers] = time.perf_counter() - start
        print(f"Benchmark: {workers} feature workers, {timings[workers]:.2f}s")
    best = min(timings, key=timings.get)
    selected = best if timings[best] < timings[0] * 0.9 else 0
    print(f"Feature workers: {selected}; measured seconds {timings}")
    return selected


def metric_totals(logs, ids, weights):
    valid = ids != 255
    safe = np.where(valid, ids, 0)
    rank = np.argsort(-logs, axis=1, kind="stable")
    order = np.lexsort((safe, -weights), axis=1)
    hard = np.take_along_axis(safe, order, axis=1)[:, 0]
    result = dict(
        examples=len(ids),
        retained_mass=float(weights.sum()),
        weighted_loss=-float((np.take_along_axis(logs, safe, axis=1) * weights).sum()),
    )
    for k in (1, 5):
        matches = (safe[:, :, None] == rank[:, None, :k]).any(axis=2) & valid
        result[f"soft_top{k}_mass"] = float((weights * matches).sum())
        result[f"hard_top{k}_correct"] = int(
            (rank[:, :k] == hard[:, None]).any(axis=1).sum()
        )
    return result


def evaluate(model, path, rows, lower, upper, batch_size=1024, workers=0):
    totals, baseline, diagnostics = {}, {}, {}
    prior = (model["prior"] + 1e-6) / (model["prior"].sum() + 148e-6)
    with prepared_batches(
        path, rows, lower, upper, batch_size, workers, diagnostics
    ) as batches:
        for features, ids, weights in batches:
            logs = log_probabilities(model, features)
            for output, values in (
                (totals, metric_totals(logs, ids, weights)),
                (
                    baseline,
                    metric_totals(
                        np.broadcast_to(np.log(prior), logs.shape), ids, weights
                    ),
                ),
            ):
                for key, value in values.items():
                    output[key] = output.get(key, 0) + value

    def finish(values):
        if not values:
            return dict(examples=0, warning="No valid adjacent targets in this split")
        result = dict(values)
        result["soft_cross_entropy"] = values["weighted_loss"] / values["retained_mass"]
        result["mean_weighted_loss"] = values["weighted_loss"] / values["examples"]
        for k in (1, 5):
            result[f"soft_top{k}"] = (
                values[f"soft_top{k}_mass"] / values["retained_mass"]
            )
            result[f"hard_top{k}_accuracy"] = (
                values[f"hard_top{k}_correct"] / values["examples"]
            )
        return result

    return dict(
        model=finish(totals),
        frequency_baseline=finish(baseline),
        diagnostics=diagnostics,
    )


def train(
    path=EVENTS_PATH,
    output=MODEL_PATH,
    epochs=3,
    batch_size=1024,
    learning_rate=0.01,
    l2=1e-4,
    workers="auto",
    limit=None,
    cutoffs=None,
    resume=False,
):
    if (
        epochs < 1
        or batch_size < 1
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
        or not np.isfinite(l2)
        or l2 < 0
    ):
        raise ValueError("Invalid training settings")
    path, output = Path(path).resolve(), Path(output).resolve()
    destinations = [
        output,
        output.with_suffix(output.suffix + ".partial"),
        output.with_suffix(".metrics.json"),
        output.with_suffix(".metrics.json.partial"),
    ]
    if any(
        candidate == path or (candidate.exists() and candidate.samefile(path))
        for candidate in destinations
    ):
        raise ValueError("Model and metrics output must differ from the source dataset")
    rows = inspect_source(path)
    rows = rows if limit is None else min(rows, limit)
    if rows < 2:
        raise ValueError("Need at least two events")
    cutoffs = time_cutoffs(path, rows) if cutoffs is None else list(cutoffs)
    if len(cutoffs) != 2 or cutoffs[0] >= cutoffs[1]:
        raise ValueError("Training cutoff must precede validation cutoff")
    stat = path.stat()
    config = dict(
        source=[str(path), stat.st_size, stat.st_mtime_ns],
        rows=rows,
        cutoffs=cutoffs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        l2=l2,
    )
    if resume:
        model = load_model(output)
        if model["metadata"].get("config") != config:
            raise ValueError("Resume source or training settings changed")
        for key in ("weights", "bias", "prior"):
            model[key] = model[key].copy()
        workers = model["metadata"]["workers"]
    else:
        if output.exists():
            raise FileExistsError("Model exists; use --resume or a new output path")
        model = new_model()
        workers = (
            choose_workers(path, rows, -np.inf, cutoffs[0], batch_size)
            if str(workers) == "auto"
            else int(workers)
        )
        model["metadata"].update(config=config, workers=workers)
    for epoch in range(model["metadata"]["epoch"], epochs):
        count = 0
        prior = np.zeros(148, dtype=np.float64)
        diagnostics = {}
        with (
            prepared_batches(
                path, rows, -np.inf, cutoffs[0], batch_size, workers, diagnostics
            ) as batches,
            tqdm(desc=f"Epoch {epoch + 1}", unit=" examples") as progress,
        ):
            for features, ids, weights in batches:
                loss = train_batch(model, features, ids, weights, learning_rate, l2)
                if not np.isfinite(loss):
                    raise ValueError(
                        "Nonfinite training objective; reduce learning rate"
                    )
                valid = ids != 255
                prior += np.bincount(ids[valid], weights=weights[valid], minlength=148)
                count += len(ids)
                progress.update(len(ids))
        if not count:
            raise ValueError("No valid training transitions")
        model["prior"] = prior
        model["metadata"]["epoch"] = epoch + 1
        model["metadata"]["training_diagnostics"] = diagnostics
        print(f"Evaluating validation split after epoch {epoch + 1}...")
        model["metadata"]["validation"] = evaluate(
            model, path, rows, cutoffs[0], cutoffs[1], batch_size, workers
        )
        save_model(model, output)
    print("Evaluating test split...")
    report = dict(
        training=model["metadata"]["training_diagnostics"],
        validation=model["metadata"]["validation"],
        test=evaluate(model, path, rows, cutoffs[1], np.inf, batch_size, workers),
    )
    report_path = output.with_suffix(".metrics.json")
    temporary = report_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(report_path)
    print(json.dumps(report, indent=2))
    return model, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=EVENTS_PATH)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument(
        "--workers", choices=["auto", "0", "1", "2", "3"], default="auto"
    )
    parser.add_argument("--train-end", help="UTC YYYY-MM-DD, exclusive")
    parser.add_argument("--validation-end", help="UTC YYYY-MM-DD, exclusive")
    parser.add_argument("--resume", action="store_true")
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument("--limit", type=int)
    size.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if bool(args.train_end) != bool(args.validation_end):
        parser.error("Provide both cutoff dates")
    cutoffs = None
    if args.train_end:
        cutoffs = [
            int(np.datetime64(day, "D").astype(np.int64)) * DAY_MS
            for day in (args.train_end, args.validation_end)
        ]
    train(
        args.events,
        args.output,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.l2,
        args.workers,
        args.limit,
        cutoffs,
        args.resume,
    )


if __name__ == "__main__":
    main()
