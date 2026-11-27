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
from scipy import sparse
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from categorical_ml import EVENTS_PATH, MAPPING_KEY, decode_top3, load_taxonomy

DAY_MS = 86_400_000
TIMESCALES = np.array([1.0, 7.0, 30.0, 180.0])
PURCHASE_SCALES = np.array([1.0, 3.0, 10.0, 30.0])
PURCHASE_DECAY = np.exp(-1 / PURCHASE_SCALES).astype(np.float32)
TIME_WEIGHT_FLOOR = 0.1
CATEGORY_COUNT = 148
FEATURE_COUNT = 740
MODEL_PATH = Path(__file__).resolve().parents[2] / "data/models/next_category_sequence.npz"


def validate_history(category_ids, category_weights):
    ids, weights = np.asarray(category_ids), np.asarray(category_weights)
    if ids.shape == (0,) and weights.shape == (0,):
        ids = np.empty((0, 3), dtype=np.uint8)
        weights = np.empty((0, 3), dtype=np.float32)
    if ids.ndim != 2 or ids.shape[1] != 3 or weights.shape != ids.shape:
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
    return ids, weights


def new_state():
    """start empty purchase histories for each sequence length"""
    return {
        "history": np.zeros((4, CATEGORY_COUNT), dtype=np.float32),
        "totals": np.zeros(CATEGORY_COUNT, dtype=np.float32),
    }


def update_state(state, ids, weights):
    # advance one purchase, no real or assumed timestamps
    state["history"] *= PURCHASE_DECAY[:, None]
    valid = ids != 255
    state["history"][:, ids[valid]] += weights[valid]
    state["totals"][ids[valid]] += weights[valid]
    return state


def training_weights(gap_days):
    """closer purchases carry more weight during learning only"""
    gap_days = np.asarray(gap_days, dtype=np.float64)
    if not np.isfinite(gap_days).all() or (gap_days < 0).any():
        raise ValueError("Training gaps must be finite and nonnegative")
    decay = np.exp(-gap_days[..., None] / TIMESCALES).mean(axis=-1)
    return (TIME_WEIGHT_FLOOR + (1 - TIME_WEIGHT_FLOOR) * decay).astype(np.float32)


def history_features(state):
    """combine recent histories and long-term category frequency"""
    total = float(state["totals"].sum())
    frequencies = state["totals"]
    if total:
        frequencies = state["totals"] / total
    return np.concatenate((state["history"].ravel(), frequencies)).astype(
        np.float32, copy=False
    )


def new_model():
    return {
        "weights": np.zeros((FEATURE_COUNT, CATEGORY_COUNT), dtype=np.float32),
        "bias": np.zeros(CATEGORY_COUNT, dtype=np.float32),
        "prior": np.zeros(CATEGORY_COUNT, dtype=np.float64),
        "metadata": {
            "version": 2,
            "feature_definition": "purchase-decay-v1",
            "purchase_scales": PURCHASE_SCALES.tolist(),
            "time_weight_floor": TIME_WEIGHT_FLOOR,
            "taxonomy": load_taxonomy(),
            "timescales": TIMESCALES.tolist(),
            "timestamp_unit": "unix_ms",
            "step": 0,
            "epoch": 0,
        },
    }


def log_probabilities(model, features):
    if sparse.issparse(features):
        product = features @ model["weights"]
    else:
        # keep single predictions from starting extra BLAS threads
        product = np.einsum("ij,jk->ik", features, model["weights"])
    logits = np.asarray(product + model["bias"], dtype=np.float64)
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite model logits")
    logits -= logits.max(axis=1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def predict_next(model, category_ids, category_weights):
    """predict next category from history, keeps model and inputs unchanged"""
    ids, weights = validate_history(category_ids, category_weights)

    # build history up to the last purchase
    state = new_state()
    for categories, values in zip(ids, weights):
        update_state(state, categories, values)
    return np.exp(log_probabilities(model, history_features(state)[None, :])[0]).astype(
        np.float32
    )


def loss_gradient(model, features, ids, weights, l2=1e-4, sample_weights=None):
    """compare predictions against all retained target weights"""
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights)
        if (
            sample_weights.shape != (len(ids),)
            or not np.isfinite(sample_weights).all()
            or (sample_weights < 0).any()
            or (sample_weights > 1).any()
        ):
            raise ValueError("Expected one finite training weight within [0, 1] per target")
        weights = weights * sample_weights[:, None]
    logs = log_probabilities(model, features)
    mass = weights.sum(axis=1)
    valid = ids != 255
    rows = np.broadcast_to(np.arange(len(ids))[:, None], ids.shape)[valid]
    labels, values = ids[valid], weights[valid]
    loss = -float(np.sum(values * logs[rows, labels])) / len(ids)

    # omitted category weight reduces this target's influence
    residual = np.exp(logs) * mass[:, None]
    np.add.at(residual, (rows, labels), -values)
    residual = (residual / len(ids)).astype(np.float32)
    gradient = np.asarray(features.T @ residual) + l2 * model["weights"]
    loss += 0.5 * l2 * float(np.square(model["weights"], dtype=np.float64).sum())
    return loss, gradient, residual.sum(axis=0)


def train_batch(
    model, features, ids, weights, learning_rate=0.01, l2=1e-4, sample_weights=None
):
    loss, gradient, bias_gradient = loss_gradient(
        model, features, ids, weights, l2, sample_weights
    )
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

    # publish only the completed model
    temporary.replace(path)


def load_model(path):
    with np.load(path, allow_pickle=False) as saved:
        model = {key: saved[key].copy() for key in ("weights", "bias", "prior")}
        model["metadata"] = json.loads(str(saved["metadata"]))
    metadata = model["metadata"]
    if (
        metadata.get("version") != 2
        or metadata.get("feature_definition") != "purchase-decay-v1"
        or metadata.get("purchase_scales") != PURCHASE_SCALES.tolist()
        or metadata.get("time_weight_floor") != TIME_WEIGHT_FLOOR
        or metadata.get("timescales") != TIMESCALES.tolist()
        or metadata.get("timestamp_unit") != "unix_ms"
        or list(metadata.get("taxonomy", {}).items()) != list(load_taxonomy().items())
    ):
        raise ValueError("Model mapping or feature definition is incompatible; retrain with next_category.py --full")
    for name, shape in (("weights", (740, 148)), ("bias", (148,)), ("prior", (148,))):
        if model[name].shape != shape or not np.isfinite(model[name]).all():
            raise ValueError("Malformed model parameters")
        # prediction callers share parameters without changing them
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

    # move each boundary so a user stays within one worker
    for partition in range(1, count):
        proposed = rows * partition // count
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
    cumulative = np.cumsum([histogram[day] for day in days])

    # split at UTC day boundaries near 80% and 90%
    cutoffs = []
    for fraction in (0.8, 0.9):
        day_index = min(np.searchsorted(cumulative, rows * fraction), len(days) - 1)
        cutoff = int((days[day_index] + 1) * DAY_MS)
        cutoffs.append(cutoff)
    if cutoffs[0] >= cutoffs[1]:
        raise ValueError(
            "Too few distinct days for automatic splits; provide explicit cutoffs"
        )
    return cutoffs


def feature_batches(path, start, stop, lower, upper, batch_size=1024, diagnostics=None):
    """build next-purchase examples one user at a time"""
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    if diagnostics is None:
        diagnostics = {}
    diagnostics.update(first_events=0, zero_mass_targets=0, examples=0)
    state, previous_user, previous_time = None, None, None
    features, labels, targets, time_weights = [], [], [], []
    for batch in read_batches(
        path, start, stop, ["user_id", "timestamp", "top3_predictions"]
    ):
        ids, weights = decode_top3(batch["top3_predictions"])
        masses = weights.sum(axis=1)
        if (masses > 1 + 3 / 510 + 1e-6).any():
            raise ValueError("Invalid retained category mass")
        timestamps = batch["timestamp"].to_numpy(zero_copy_only=False)
        users = batch["user_id"].to_pylist()

        # vectorize timing weights, first events never become targets
        gaps = np.zeros(len(batch), dtype=np.float64)
        gaps[1:] = timestamps[1:].astype(np.float64) - timestamps[:-1].astype(np.float64)
        if previous_time is not None:
            gaps[0] = int(timestamps[0]) - previous_time
        batch_weights = training_weights(np.maximum(gaps, 0) / DAY_MS)
        for user, timestamp, categories, values, mass, time_weight in zip(
            users, timestamps, ids, weights, masses, batch_weights
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
                if not same_user:
                    key = "first_events"
                elif mass > 0:
                    key = "examples"
                else:
                    key = "zero_mass_targets"
                diagnostics[key] += 1
            if not same_user:
                state = new_state()
            elif lower <= timestamp < upper and mass > 0:
                # features use past categories, timing only weights the target
                features.append(history_features(state))
                labels.append(categories)
                targets.append(values)
                time_weights.append(time_weight)
            update_state(state, categories, values)
            previous_user, previous_time = user, timestamp
            if len(features) == batch_size:
                yield (
                    sparse.csr_matrix(np.asarray(features)),
                    np.asarray(labels),
                    np.asarray(targets),
                    np.asarray(time_weights, dtype=np.float32),
                )
                features, labels, targets, time_weights = [], [], [], []
    if features:
        yield (
            sparse.csr_matrix(np.asarray(features)),
            np.asarray(labels),
            np.asarray(targets),
            np.asarray(time_weights, dtype=np.float32),
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
    """prepare bounded batches with optional worker processes"""
    if diagnostics is None:
        diagnostics = {}
    diagnostics.update(first_events=0, zero_mass_targets=0, examples=0)
    if not 0 <= workers <= 3:
        raise ValueError(
            "Use 0 through 3 feature workers, reserving a core for training"
        )
    if workers == 0:
        with threadpool_limits(limits=4):
            yield feature_batches(path, 0, rows, lower, upper, batch_size, diagnostics)
        return

    # workers prepare batches, parent trains the model
    context = get_context("spawn")
    queues, processes = [], []
    try:
        for start, stop in user_ranges(path, rows, workers):
            # bound queued batches to keep memory down
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

    # compare worker counts, include batch transfer and training
    for workers in range(4):
        model = new_model()
        start = time.perf_counter()
        with prepared_batches(
            path, min(rows, 500000), lower, upper, batch_size, workers
        ) as batches:
            for features, ids, weights, time_weights in batches:
                train_batch(model, features, ids, weights, sample_weights=time_weights)
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
    result = {
        "examples": len(ids),
        "retained_mass": float(weights.sum()),
        "weighted_loss": -float((np.take_along_axis(logs, safe, axis=1) * weights).sum()),
    }
    for top_count in (1, 5):
        matches = (safe[:, :, None] == rank[:, None, :top_count]).any(axis=2) & valid
        result[f"soft_top{top_count}_mass"] = float((weights * matches).sum())
        result[f"hard_top{top_count}_correct"] = int(
            (rank[:, :top_count] == hard[:, None]).any(axis=1).sum()
        )
    return result


def evaluate(model, path, rows, lower, upper, batch_size=1024, workers=0):
    totals, baseline, diagnostics = {}, {}, {}
    prior = (model["prior"] + 1e-6) / (model["prior"].sum() + 148e-6)
    with prepared_batches(
        path, rows, lower, upper, batch_size, workers, diagnostics
    ) as batches:
        for features, ids, weights, time_weights in batches:
            # evaluate every next purchase without timing weights
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
            return {"examples": 0, "warning": "No valid adjacent targets in this split"}
        result = values.copy()
        result["soft_cross_entropy"] = values["weighted_loss"] / values["retained_mass"]
        result["mean_weighted_loss"] = values["weighted_loss"] / values["examples"]
        for top_count in (1, 5):
            result[f"soft_top{top_count}"] = (
                values[f"soft_top{top_count}_mass"] / values["retained_mass"]
            )
            result[f"hard_top{top_count}_accuracy"] = (
                values[f"hard_top{top_count}_correct"] / values["examples"]
            )
        return result

    return {
        "model": finish(totals),
        "frequency_baseline": finish(baseline),
        "diagnostics": diagnostics,
    }


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
    """train by epoch, save completed progress for resuming"""
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
    config = {
        "source": [str(path), stat.st_size, stat.st_mtime_ns],
        "rows": rows,
        "cutoffs": cutoffs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "l2": l2,
    }
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
        if str(workers) == "auto":
            workers = choose_workers(path, rows, -np.inf, cutoffs[0], batch_size)
        else:
            workers = int(workers)
        model["metadata"].update(config=config, workers=workers)

    # resume from the last completed epoch
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
            for features, ids, weights, time_weights in batches:
                loss = train_batch(
                    model, features, ids, weights, learning_rate, l2, time_weights
                )
                if not np.isfinite(loss):
                    raise ValueError(
                        "Nonfinite training objective; reduce learning rate"
                    )
                valid = ids != 255
                weighted_targets = weights * time_weights[:, None]
                prior += np.bincount(ids[valid], weights=weighted_targets[valid], minlength=148)
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
    report = {
        "training": model["metadata"]["training_diagnostics"],
        "validation": model["metadata"]["validation"],
        "test": evaluate(model, path, rows, cutoffs[1], np.inf, batch_size, workers),
    }

    # save validation and test results beside the model
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
