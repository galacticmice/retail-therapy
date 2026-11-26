# Title-based product categorization

The cleaned events contain 75,603,822 events and no product names. This workflow
joins product metadata by `parent_asin`, classifies each distinct title once, and
optionally writes a new enriched events file. **The original
`data/processed/events_dataset.parquet` is read-only throughout.** Existing output
files cannot be overwritten. The original README.md is unchanged.

## Install and run

Activate the project environment, then install the optional dependency:

```sh
python -m pip install ".[categorization]"
```

Obtain product metadata with `parent_asin` and `title` from
[Amazon Reviews 2023 metadata](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw/meta_categories).
Use product titles, not review titles. Local JSONL/JSONL.gz files and direct HTTP(S)
URLs are supported. Multiple sources can follow `--metadata`. Full metadata is
large (about 101 GB uncompressed); URL imports stream it without saving the full
files and retain only titles for indexed products.

From the repository root:

```sh
python src/retail-therapy/categorical_ml.py titles \
  --metadata /path/to/meta_*.jsonl
python src/retail-therapy/categorical_ml.py classify
# Optional: create a NEW enriched events dataset.
python src/retail-therapy/categorical_ml.py enrich
```

For example, pass this direct URL as a metadata source for beauty products:
`https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_All_Beauty.jsonl`.
Use all relevant departments for broad coverage; missing titles remain Unknown.

Default artifacts under `data/derived/title_categories/`:

- `products.sqlite3`: disk-backed, resumable product/title/prediction cache.
- `product_labels.parquet`: one row per product, joinable by `parent_asin`.
- `events_fine_categories.parquet`: optional enriched events retaining all original
  columns, broad `category`, event count, and row order.

## Pilot and resume

```sh
python src/retail-therapy/categorical_ml.py titles \
  --work-dir data/derived/pilot --event-limit 10000 \
  --metadata /path/to/meta_All_Beauty.jsonl
python src/retail-therapy/categorical_ml.py classify \
  --work-dir data/derived/pilot --limit 100 \
  --output data/derived/pilot_labels.parquet
```

Pass the same `--work-dir` to indexing and classification. `--output` is independent
of `--work-dir`. To continue after exporting a pilot, choose a new output filename;
completed predictions are reused. A pilot covers only the indexed events; use a
new work directory for the full run. Re-running `titles` fills missing names,
retaining the first nonempty title per product. Interrupted metadata imports
rescan each input file from the beginning. Classification commits every batch.
Enrichment streams events, but the product mapping still requires join memory.

## Model and quality

The baseline uses [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
title embeddings and cosine similarity to descriptions in
[`product_taxonomy.json`](src/retail-therapy/product_taxonomy.json). It compares all
fine labels regardless of existing broad category, so Unknown items can be
recovered. The editable taxonomy is a starting vocabulary, not an exhaustive
retail ontology. Media labels distinguish formats rather than genres that may
be impossible to infer from names. Long titles are truncated to the model limit.

This is a pretrained semantic similarity model, **not a retail classifier trained
or validated on these products**. Each mapping row includes `product_title`,
`fine_category`, `suggested_category`, `similarity`, `margin`, and `label_status`.
Low similarity or a close second choice produces `fine_category=Unknown` with
`label_status=needs_review`; missing titles and pending pilot rows are marked
separately. Similarity is **not a calibrated probability**. Defaults
(`--min-similarity 0.35`, `--min-margin 0.05`) are provisional. Unsupported products
can still get incorrect accepted labels. Manually label a representative sample,
including formerly Unknown items, and measure accuracy and coverage before using
these predictions downstream. Adjust the taxonomy and thresholds accordingly.

The first inference downloads model weights; subsequent inference runs locally.
Select `--device cpu`, `mps`, or `cuda` as needed (default: automatic). `--revision`
can pin a model commit. Model settings, taxonomy, and source fingerprint are
embedded in the label mapping. Changed settings or indexed source require a new
work directory to avoid mixing incompatible predictions.

## Verification

```sh
python -m unittest discover -s tests -v
```

Tests use temporary fixture files to check source immutability, metadata imports,
resumption, missing titles, uncertainty thresholds, output alias protection,
interrupted writes, duplicate joins, and event preservation.
