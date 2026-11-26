from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import importlib
import json
from pathlib import Path
import sys
import tempfile
from threading import Thread
import unittest
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq

# Spawned workers must be able to import the module by its normal name.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/retail-therapy"))
pipeline = importlib.import_module("dataset")


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def response_for(products):
    response = MagicMock()
    response.__enter__.return_value = response
    response.iter_lines.return_value = [json.dumps(p).encode() for p in products]
    return response


class DatasetTitleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("prep", "raw", "processed"):
            path = self.root / name
            path.mkdir()
            patcher = patch.object(pipeline, name.upper() + "FOLDER_PATH", path)
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_events(self, category, ids):
        rows = [dict(user_id="u1", parent_asin=asin, category=category, timestamp=i)
                for i, asin in enumerate(ids)]
        path = self.root / "raw" / f"{category}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        return path

    def test_matching_titles_only_deduplicated_and_cached(self):
        source = self.write_events("Electronics", ["A", "A", "B", "C"])
        before = source.read_bytes()
        products = [dict(parent_asin=a, title=t) for a,t in
                    [("Z", "Outside cohort"), ("A", " Keyboard "),
                     ("A", "Duplicate title"), ("B", ""), ("B", "Mouse")]]
        with patch.object(pipeline.requests, "get", return_value=response_for(products)) as get:
            output = pipeline.import_category_titles(source)
            self.assertEqual(pipeline.import_category_titles(source), output)
            get.assert_called_once_with(
                pipeline.METADATA_URL + "/meta_Electronics.jsonl", stream=True, timeout=120
            )
        rows = pq.read_table(output).to_pylist()
        self.assertEqual({r["parent_asin"]: r["title"] for r in rows}, {"A":"Keyboard", "B":"Mouse"})
        self.assertEqual(source.read_bytes(), before)
        # A changed raw file invalidates the title cache.
        self.write_events("Electronics", ["A", "D"])
        with patch.object(pipeline.requests, "get", return_value=response_for([
            dict(parent_asin="A", title="Keyboard"), dict(parent_asin="D", title="Monitor")
        ])) as get:
            pipeline.import_category_titles(source)
            get.assert_called_once()
        self.assertEqual(set(pq.read_table(output)["parent_asin"].to_pylist()), {"A", "D"})

    def test_merge_retains_all_events_original_fields_and_missing_titles(self):
        originals = []
        for index in range(34):
            source = self.write_events(f"Category_{index:02}", ["A", "A", "missing"])
            originals.extend(pq.read_table(source).to_pylist())
        # Exercise real spawned processes against a local fixture HTTP server.
        metadata_dir = self.root / "metadata"
        metadata_dir.mkdir()
        for index in range(34):
            (metadata_dir / f"meta_Category_{index:02}.jsonl").write_text(
                json.dumps(dict(parent_asin="A", title="A product")) + "\n"
            )
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), partial(QuietHandler, directory=str(metadata_dir))
        )
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            with patch.object(pipeline, "METADATA_URL", url), patch("builtins.input", return_value=""), patch.object(
                pipeline, "ProcessPoolExecutor", wraps=pipeline.ProcessPoolExecutor
            ) as pool:
                pipeline.launch_parquet_combine()
                # The second merge reuses every checkpoint and starts no new pool.
                pipeline.launch_parquet_combine()
                pool.assert_called_once()
                self.assertEqual(pool.call_args.kwargs["max_workers"], 4)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        rows = pq.read_table(self.root / "processed/events_dataset.parquet").to_pylist()
        self.assertEqual(len(rows), len(originals))
        fields = ["user_id", "parent_asin", "category", "timestamp"]
        self.assertCountEqual([tuple(r[k] for k in fields) for r in rows],
                              [tuple(r[k] for k in fields) for r in originals])
        self.assertEqual([r["timestamp"] for r in rows], sorted(r["timestamp"] for r in rows))
        for row in rows:
            self.assertEqual(row["title"], None if row["parent_asin"] == "missing" else "A product")

    def test_existing_filtered_events_reused_without_sorting_or_raw_rows(self):
        source = self.root / "processed/events_dataset.parquet"
        original = [
            dict(user_id="u1", parent_asin="A", category="Electronics", timestamp=1,
                 context="first", title="old title"),
            dict(user_id="u1", parent_asin="A", category="Electronics", timestamp=1,
                 context="second", title="old title"),
            dict(user_id="u2", parent_asin="missing", category="Electronics", timestamp=2,
                 context="third", title=None),
        ]
        pq.write_table(pa.Table.from_pylist(original), source)
        # This extra raw product must neither enter the events nor the title cache.
        self.write_events("Electronics", ["A", "excluded"])
        metadata = response_for([
            dict(parent_asin="A", title="Keyboard"),
            dict(parent_asin="excluded", title="Outside filtered events"),
        ])

        def run_workers(directory, func, text, **kwargs):
            for item in kwargs["items"].items():
                if not kwargs["is_complete"](item):
                    func(item)

        with patch.object(pipeline, "launch_parallel_func", side_effect=run_workers), patch.object(
            pipeline.requests, "get", return_value=metadata
        ), patch("builtins.input", return_value=""), patch.object(
            pipeline.pl.LazyFrame, "sort", side_effect=AssertionError("must not sort existing events")
        ), patch.object(pipeline.pl, "scan_parquet", wraps=pipeline.pl.scan_parquet) as scan:
            pipeline.launch_parquet_combine()
        self.assertTrue(all(str(self.root / "raw") not in str(call) for call in scan.call_args_list))
        rows = pq.read_table(source).to_pylist()
        self.assertEqual([{k:v for k,v in row.items() if k != "title"} for row in rows],
                         [{k:v for k,v in row.items() if k != "title"} for row in original])
        self.assertEqual([row["title"] for row in rows], ["Keyboard", "Keyboard", None])
        self.assertNotIn("title_right", rows[0])
        titles = pq.read_table(self.root / "prep/titles/Electronics.parquet").to_pylist()
        self.assertEqual([row["parent_asin"] for row in titles], ["A"])

    def test_failed_enrichment_leaves_existing_events_unchanged(self):
        source = self.root / "processed/events_dataset.parquet"
        pq.write_table(pa.Table.from_pylist([
            dict(user_id="u1", parent_asin="A", category="Electronics", timestamp=1)
        ]), source)
        before = source.read_bytes()
        with patch.object(pipeline, "launch_parallel_func", side_effect=RuntimeError("worker failed")):
            with self.assertRaises(RuntimeError):
                pipeline.launch_parquet_combine()
        self.assertEqual(source.read_bytes(), before)

    def test_shutdown_prevents_title_download(self):
        source = self.write_events("Electronics", ["A"])
        event = MagicMock()
        event.is_set.return_value = True
        with patch.object(pipeline, "_stop_event", event), patch.object(pipeline.requests, "get") as get:
            with self.assertRaises(pipeline.ShutdownRequested):
                pipeline.import_category_titles(source)
            get.assert_not_called()

    def test_failed_metadata_request_does_not_publish_cache(self):
        source = self.write_events("Electronics", ["A"])
        response = response_for([])
        response.raise_for_status.side_effect = pipeline.requests.HTTPError("failed download")
        with patch.object(pipeline.requests, "get", return_value=response):
            with self.assertRaises(pipeline.requests.HTTPError):
                pipeline.import_category_titles(source)
        self.assertFalse((self.root / "prep/titles/Electronics.parquet").exists())


if __name__ == "__main__":
    unittest.main()
