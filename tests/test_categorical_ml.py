import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location("ml", Path(__file__).resolve().parents[1] / "src/retail-therapy/categorical_ml.py")
ml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml)


class CategorizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.events = self.root / "events_dataset.parquet"
        self.rows = [dict(user_id=u, parent_asin=a, category=c, timestamp=t) for u,a,c,t in
                     [("u1","A","Unknown",1), ("u1","B","Electronics",2),
                      ("u2","A","Unknown",3), ("u2","C","Unknown",4)]]
        pq.write_table(pa.Table.from_pylist(self.rows), self.events)
        self.before = ml.digest(self.events)
        self.work = self.root / "work"

    def tearDown(self):
        self.assertEqual(ml.digest(self.events), self.before)

    def prepare(self):
        meta = self.root / "meta.jsonl.gz"
        with gzip.open(meta, "wt") as stream:
            for asin,title in [("A"," Mechanical   keyboard "), ("B"," "), ("Z","Other")]:
                stream.write(json.dumps(dict(parent_asin=asin, title=title)) + "\n")
        args = argparse.Namespace(events=self.events, work_dir=self.work, metadata=[str(meta)], event_limit=None)
        ml.titles(args)
        return args

    def test_import_resume_and_missing_titles(self):
        ml.titles(self.prepare())
        with ml.connect(self.work, self.events) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM products").fetchone()[0], 3)
            self.assertEqual(db.execute("SELECT title FROM products WHERE parent_asin='A'").fetchone()[0], "Mechanical keyboard")
            output = self.root / "labels.parquet"
            ml.export_labels(db, output, self.events, self.work)
        self.assertEqual({r["parent_asin"]: r["label_status"] for r in pq.read_table(output).to_pylist()},
                         {"A":"pending", "B":"missing_title", "C":"missing_title"})

    def test_preserves_every_original_event_and_order(self):
        self.prepare()
        labels = self.root / "labels.parquet"
        with ml.connect(self.work, self.events) as db:
            db.execute("UPDATE products SET fine_category='Keyboard', label_status='accepted' WHERE parent_asin='A'")
            ml.export_labels(db, labels, self.events, self.work)
        output = self.root / "enriched.parquet"
        ml.enrich(argparse.Namespace(events=self.events, labels=labels, output=output))
        rows = pq.read_table(output).to_pylist()
        self.assertEqual([{k:r[k] for k in self.rows[0]} for r in rows], self.rows)
        self.assertEqual([r["fine_category"] for r in rows], ["Keyboard","Unknown","Keyboard","Unknown"])

    def test_source_aliases_and_existing_outputs_refused(self):
        with self.assertRaises(ValueError):
            ml.safe_target(self.events)
        alias = self.root / "alias.parquet"
        alias.symlink_to(self.events)
        with self.assertRaises(ValueError):
            ml.safe_target(alias, self.events)
        alias.unlink()
        alias.hardlink_to(self.events)
        with self.assertRaises(ValueError):
            ml.safe_target(alias, self.events)
        existing = self.root / "existing.parquet"
        existing.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            with ml.new_output(existing):
                pass
        self.assertEqual(existing.read_bytes(), b"keep")

    def test_failed_and_racing_outputs_do_not_overwrite(self):
        output = self.root / "new.parquet"
        with self.assertRaises(RuntimeError):
            with ml.new_output(output) as temporary:
                temporary.write_bytes(b"incomplete")
                raise RuntimeError("interrupted")
        self.assertFalse(output.exists())
        with self.assertRaises(FileExistsError):
            with ml.new_output(output) as temporary:
                temporary.write_bytes(b"ours")
                output.write_bytes(b"other")
        self.assertEqual(output.read_bytes(), b"other")

    def test_duplicate_mapping_does_not_publish(self):
        import polars as pl
        labels = self.root / "duplicates.parquet"
        row = dict(zip(ml.LABEL_SCHEMA.names, ["A","Keyboard","Keyboard","Keyboard",.8,.4,"accepted"]))
        pq.write_table(pa.Table.from_pylist([row,row], schema=ml.LABEL_SCHEMA), labels)
        output = self.root / "invalid.parquet"
        with self.assertRaises(pl.exceptions.ComputeError):
            ml.enrich(argparse.Namespace(events=self.events, labels=labels, output=output))
        self.assertFalse(output.exists())

    def test_thresholds_and_settings_guard(self):
        labels = ["Keyboard","Shampoo"]
        self.assertEqual(ml.decide(labels, [.8,.2], .35,.05)[0], "Keyboard")
        self.assertEqual(ml.decide(labels, [.31,.1], .35,.05)[4], "needs_review")
        self.assertEqual(ml.decide(labels, [.8,.79], .35,.05)[0], "Unknown")
        self.prepare()
        with ml.connect(self.work, self.events) as db:
            ml.pin(db, "classifier", {"model":"one"})
            with self.assertRaises(ValueError):
                ml.pin(db, "classifier", {"model":"two"})


if __name__ == "__main__":
    unittest.main()
