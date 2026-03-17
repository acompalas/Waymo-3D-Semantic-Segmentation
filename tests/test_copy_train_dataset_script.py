import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "copy_train_dataset.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("copy_train_dataset_script", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _build_fake_dataset(root: Path, dataset_name: str) -> None:
    dataset_root = root / dataset_name
    _write_json(dataset_root / "meta.json", {"representation": dataset_name})
    _write_json(dataset_root / "classes.json", {"0": "undefined", "1": "car"})
    _write_json(dataset_root / "extra_metadata.json", {"copied_from": "unit-test"})
    _write_json(
        dataset_root / "segment_source.json",
        [
            {"source_subdir": "training", "is_labeled": True, "segments": ["segment_train_a", "segment_train_b"]},
            {"source_subdir": "validation", "is_labeled": True, "segments": ["segment_val"]},
        ],
    )

    for segment_name, payload in {
        "segment_train_a": "train-a",
        "segment_train_b": "train-b",
        "segment_val": "val",
    }.items():
        segment_dir = dataset_root / "segments" / segment_name
        segment_dir.mkdir(parents=True, exist_ok=True)
        (segment_dir / "payload.txt").write_text(payload, encoding="utf-8")


class CopyTrainDatasetScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tmpdir.name)
        self.module = _load_script_module()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_copy_keeps_only_training_segments_for_supported_datasets(self) -> None:
        source_root = self.base_dir / "source"
        dest_root = self.base_dir / "dest"

        for dataset_name in ("point_clouds", "range_images"):
            _build_fake_dataset(source_root, dataset_name)
            copied_dir = self.module.copy_train_split_dataset(
                dataset=dataset_name,
                source_root=source_root,
                dest_root=dest_root,
            )

            self.assertEqual(copied_dir, dest_root / dataset_name)
            self.assertTrue((copied_dir / "meta.json").exists())
            self.assertTrue((copied_dir / "classes.json").exists())
            self.assertTrue((copied_dir / "extra_metadata.json").exists())

            copied_segments = sorted(path.name for path in (copied_dir / "segments").iterdir() if path.is_dir())
            self.assertEqual(copied_segments, ["segment_train_a", "segment_train_b"])
            self.assertFalse((copied_dir / "segments" / "segment_val").exists())
            self.assertEqual((copied_dir / "segments" / "segment_train_a" / "payload.txt").read_text(), "train-a")
            self.assertEqual((copied_dir / "segments" / "segment_train_b" / "payload.txt").read_text(), "train-b")

            copied_segment_source = json.loads((copied_dir / "segment_source.json").read_text(encoding="utf-8"))
            self.assertEqual(
                copied_segment_source,
                [
                    {
                        "source_subdir": "training",
                        "is_labeled": True,
                        "segments": ["segment_train_a", "segment_train_b"],
                    },
                    {"source_subdir": "validation", "is_labeled": True, "segments": []},
                ],
            )


if __name__ == "__main__":
    unittest.main()
