"""Title-based product labels. The source events Parquet is always read-only."""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from urllib.request import urlopen

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS = ROOT / "data/processed/events_dataset.parquet"
DEFAULT_WORK = ROOT / "data/derived/title_categories"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_target(target, *sources, must_be_new=True):
    target = Path(target).absolute()
    if target.name == "events_dataset.parquet":
        raise ValueError("events_dataset.parquet is protected; choose a different output name")
    for source in sources:
        source = Path(source)
        if target.resolve() == source.resolve() or (
            target.exists() and source.exists() and os.path.samefile(target, source)
        ):
            raise ValueError(f"Output aliases a protected input: {source}")
    if target.is_symlink() or (must_be_new and target.exists()):
        raise FileExistsError(f"Refusing to overwrite {target}; choose a new output path")
    return target


@contextmanager
def new_output(target, *sources):
    """Write privately, then publish without replacing any existing file."""
    target = safe_target(target, DEFAULT_EVENTS, *sources)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    try:
        yield Path(temporary)
        os.link(temporary, target)  # Atomic, and fails if target appeared meanwhile.
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def connect(work, events, *, create=False):
    db_path = safe_target(Path(work) / "products.sqlite3", events, DEFAULT_EVENTS,
                          must_be_new=False)
    if not create and not db_path.exists():
        raise ValueError("Run titles first to build the product index")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT)")
    db.execute("""CREATE TABLE IF NOT EXISTS products (
        parent_asin TEXT PRIMARY KEY, title TEXT, fine_category TEXT,
        suggested_category TEXT, similarity REAL, margin REAL, label_status TEXT)""")
    db.execute("""CREATE INDEX IF NOT EXISTS pending_titles ON products(parent_asin)
                  WHERE title IS NOT NULL AND label_status IS NULL""")
    try:
        with db:
            yield db
    finally:
        db.close()


def pin(db, key, value):
    value = json.dumps(value, sort_keys=True)
    old = db.execute("SELECT value FROM config WHERE key=?", (key,)).fetchone()
    if old and old[0] != value:
        raise ValueError(f"{key} changed; use a new --work-dir to avoid stale labels")
    db.execute("INSERT OR IGNORE INTO config VALUES (?, ?)", (key, value))
    db.commit()


@contextmanager
def metadata_lines(location):
    """Read local/HTTP JSONL or JSONL.gz without downloading a complete file."""
    if location.startswith(("https://", "http://")):
        stream = urlopen(location, timeout=120)
    else:
        stream = open(location, "rb")
    with stream:
        if location.split("?", 1)[0].endswith(".gz"):
            with gzip.GzipFile(fileobj=stream) as decompressed:
                yield decompressed
        else:
            yield stream


def import_titles(db, location):
    pending = []
    def flush():
        with db:
            db.executemany("""UPDATE products SET title=? WHERE parent_asin=?
                              AND title IS NULL""", pending)
        pending.clear()
    with metadata_lines(location) as lines:
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError(f"Invalid JSON in {location}, line {number}") from error
            title, asin = row.get("title"), row.get("parent_asin")
            if isinstance(title, str) and title.strip() and isinstance(asin, str):
                pending.append((" ".join(title.split()), asin))
            if number % 10000 == 0:
                flush()
                print(f"Metadata rows scanned: {number:,}", flush=True)
        flush()


def titles(args):
    with connect(args.work_dir, args.events, create=True) as db:
        pin(db, "source", {"path": str(args.events.resolve()), "sha256": digest(args.events),
                           "event_limit": args.event_limit})
        if not db.execute("SELECT 1 FROM config WHERE key='indexed'").fetchone():
            seen = 0
            for batch in pq.ParquetFile(args.events).iter_batches(
                columns=["parent_asin"], batch_size=100000
            ):
                ids = batch.column(0).to_pylist()
                if args.event_limit is not None:
                    ids = ids[:max(0, args.event_limit - seen)]
                with db:
                    db.executemany("INSERT OR IGNORE INTO products(parent_asin) VALUES (?)",
                                   ((asin,) for asin in ids if asin))
                seen += len(ids)
                print(f"Events indexed: {seen:,}", flush=True)
                if args.event_limit is not None and seen >= args.event_limit:
                    break
            pin(db, "indexed", True)
        for location in args.metadata:
            print(f"Reading {location}", flush=True)
            import_titles(db, location)
        total, matched = db.execute(
            "SELECT count(*), count(title) FROM products"
        ).fetchone()
        print(f"Titles found: {matched:,} / {total:,} distinct products")


def load_taxonomy(path):
    taxonomy = json.loads(Path(path).read_text())
    if not isinstance(taxonomy, dict) or len(taxonomy) < 2:
        raise ValueError("Taxonomy must map at least two fine labels to descriptions")
    if any(not isinstance(k, str) or not k.strip() or k == "Unknown" or
           not isinstance(v, str) or not v.strip() for k, v in taxonomy.items()):
        raise ValueError("Taxonomy labels/descriptions must be nonempty strings; Unknown is reserved")
    return taxonomy


def decide(labels, scores, min_similarity, min_margin):
    ranked = sorted(range(len(labels)), key=lambda i: float(scores[i]), reverse=True)
    best, runner_up = ranked[:2]
    score = float(scores[best])
    margin = score - float(scores[runner_up])
    accepted = score >= min_similarity and margin >= min_margin
    return (labels[best] if accepted else "Unknown", labels[best], score, margin,
            "accepted" if accepted else "needs_review")


def classify(args):
    taxonomy = load_taxonomy(args.taxonomy)
    with connect(args.work_dir, args.events) as db:
        pin(db, "classifier", {"model": args.model, "revision": args.revision,
            "taxonomy": taxonomy, "min_similarity": args.min_similarity,
            "min_margin": args.min_margin})
        remaining = db.execute("""SELECT count(*) FROM products
            WHERE title IS NOT NULL AND label_status IS NULL""").fetchone()[0]
        if remaining:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise SystemExit('Install ML dependencies: python -m pip install ".[categorization]"') from error
            model = SentenceTransformer(args.model, revision=args.revision, device=args.device)
            labels = list(taxonomy)
            embeddings = model.encode(list(taxonomy.values()), normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False)
            done = 0
            while args.limit is None or done < args.limit:
                size = args.batch_size if args.limit is None else min(args.batch_size, args.limit - done)
                rows = db.execute("""SELECT parent_asin, title FROM products
                    WHERE title IS NOT NULL AND label_status IS NULL
                    ORDER BY parent_asin LIMIT ?""", (size,)).fetchall()
                if not rows:
                    break
                vectors = model.encode([r[1] for r in rows], normalize_embeddings=True,
                                       convert_to_numpy=True, batch_size=args.batch_size,
                                       show_progress_bar=False)
                scores = vectors @ embeddings.T
                updates = [(*decide(labels, score, args.min_similarity, args.min_margin), row[0])
                           for row, score in zip(rows, scores)]
                with db:
                    db.executemany("""UPDATE products SET fine_category=?, suggested_category=?,
                        similarity=?, margin=?, label_status=? WHERE parent_asin=?""", updates)
                done += len(rows)
                print(f"Classified {done:,} / {remaining:,} pending products", flush=True)
        export_labels(db, args.output, args.events, args.work_dir)


LABEL_SCHEMA = pa.schema([
    ("parent_asin", pa.string()), ("product_title", pa.string()),
    ("fine_category", pa.string()), ("suggested_category", pa.string()),
    ("similarity", pa.float64()), ("margin", pa.float64()), ("label_status", pa.string())
])


def export_labels(db, output, events, work):
    with new_output(output, events, Path(work) / "products.sqlite3") as temporary:
        config = dict(db.execute("SELECT key, value FROM config"))
        schema = LABEL_SCHEMA.with_metadata({k.encode(): v.encode() for k, v in config.items()})
        cursor = db.execute("""SELECT parent_asin, title, coalesce(fine_category, 'Unknown'),
            suggested_category, similarity, margin,
            coalesce(label_status, CASE WHEN title IS NULL THEN 'missing_title' ELSE 'pending' END)
            FROM products""")
        with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
            while rows := cursor.fetchmany(10000):
                writer.write_table(pa.Table.from_pylist(
                    [dict(zip(schema.names, row)) for row in rows], schema=schema))
    print(f"New label mapping: {output}")
    print(dict(db.execute("""SELECT coalesce(label_status,
        CASE WHEN title IS NULL THEN 'missing_title' ELSE 'pending' END), count(*)
        FROM products GROUP BY 1""")))


def enrich(args):
    import polars as pl
    safe_target(args.output, args.events, args.labels, DEFAULT_EVENTS)
    events = pl.scan_parquet(args.events)
    labels = pl.scan_parquet(args.labels)
    required = set(LABEL_SCHEMA.names)
    if not required <= set(labels.collect_schema().names()):
        raise ValueError("Labels file does not contain the expected mapping columns")
    overlap = (set(events.collect_schema().names()) & required) - {"parent_asin"}
    if overlap:
        raise ValueError(f"Events already contain enrichment columns: {sorted(overlap)}")
    with new_output(args.output, args.events, args.labels) as temporary:
        (events.join(labels.select(LABEL_SCHEMA.names), on="parent_asin", how="left",
                     validate="m:1", maintain_order="left")
         .with_columns(pl.col("fine_category").fill_null("Unknown"),
                       pl.col("label_status").fill_null("missing_metadata"))
         .sink_parquet(temporary, compression="zstd", engine="streaming", maintain_order=True))
        if pq.ParquetFile(temporary).metadata.num_rows != pq.ParquetFile(args.events).metadata.num_rows:
            raise ValueError("Event count changed; refusing to publish output")
    print(f"New enriched events: {args.output}")


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("titles", help="Index products and import Amazon product metadata")
    prep.add_argument("--metadata", nargs="+", required=True, help="Local paths or URLs to JSONL[.gz]")
    prep.add_argument("--event-limit", type=positive, help="Pilot: index only the first N events")
    prep.set_defaults(func=titles)
    predict = sub.add_parser("classify", help="Embed titles, assign fine labels, export new mapping")
    predict.add_argument("--taxonomy", type=Path, default=Path(__file__).with_name("product_taxonomy.json"))
    predict.add_argument("--model", default=DEFAULT_MODEL)
    predict.add_argument("--revision", help="Pin a Hugging Face model commit for reproducibility")
    predict.add_argument("--device", help="cpu, mps, or cuda; auto-detected by default")
    predict.add_argument("--min-similarity", type=float, default=0.35)
    predict.add_argument("--min-margin", type=float, default=0.05)
    predict.add_argument("--batch-size", type=positive, default=128)
    predict.add_argument("--limit", type=positive, help="Classify at most N pending products this run")
    predict.add_argument("--output", type=Path, default=DEFAULT_WORK / "product_labels.parquet")
    predict.set_defaults(func=classify)
    merge = sub.add_parser("enrich", help="Left join labels into a NEW events file")
    merge.add_argument("--labels", type=Path, default=DEFAULT_WORK / "product_labels.parquet")
    merge.add_argument("--output", type=Path, default=DEFAULT_WORK / "events_fine_categories.parquet")
    merge.set_defaults(func=enrich)
    for command in (prep, predict, merge):
        command.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
        command.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()
    if args.command == "classify":
        if not (-1 <= args.min_similarity <= 1 and 0 <= args.min_margin <= 2):
            parser.error("similarity must be in [-1, 1] and margin in [0, 2]")
        safe_target(args.output, args.events, DEFAULT_EVENTS, Path(args.work_dir) / "products.sqlite3")
    args.func(args)


if __name__ == "__main__":
    main()
