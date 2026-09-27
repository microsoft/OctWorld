"""Offline downloader regressions: python -m unittest discover -s tests -v."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import download_models as downloader


def entry_for(data: bytes, name: str = "model.pt", algorithm: str = "sha256") -> list:
    digest = (hashlib.sha256(data) if algorithm == "sha256" else
              hashlib.sha1(f"blob {len(data)}\0".encode() + data))
    return [name, len(data), algorithm, digest.hexdigest()]


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = b"test model bytes"
        self.entry = entry_for(self.data)
        self.model = {"directory": "moge", "url": "https://example.invalid/weights", "files": [self.entry]}

    def run_main(self, *args, manifest=None):
        with patch.object(downloader, "load_manifest", return_value=manifest or {"moge": self.model}), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return downloader.main(["--output-dir", str(self.root / "weights"), *args])

    def test_manifest_has_exact_public_components_and_complete_qwen(self):
        manifest = downloader.load_manifest()
        self.assertEqual(set(manifest), {"wan", "da3", "raft", "skyseg", "moge", "qwen"})
        names = {entry[0] for entry in manifest["qwen"]["files"]}
        self.assertIn("preprocessor_config.json", names)
        self.assertIn("video_preprocessor_config.json", names)
        self.assertIn("chat_template.json", names)
        self.assertNotIn("processor_config.json", names)
        self.assertEqual(sum(name.endswith(".safetensors") for name in names), 4)
        self.assertNotIn("diffusion_pytorch_model.safetensors", {e[0] for e in manifest["wan"]["files"]})

    def test_sha256_and_git_blob_hashes(self):
        path = self.root / "model"
        path.write_bytes(self.data)
        for algorithm in ("sha256", "git-sha1"):
            self.assertIsNone(downloader.file_error(path, entry_for(self.data, algorithm=algorithm)))

    def test_missing_truncated_and_corrupt_files(self):
        path = self.root / "model"
        self.assertEqual(downloader.file_error(path, self.entry), "missing file")
        path.write_bytes(b"short")
        self.assertIn("size mismatch", downloader.file_error(path, self.entry))
        path.write_bytes(b"x" * len(self.data))
        self.assertEqual(downloader.file_error(path, self.entry), "sha256 mismatch")

    def test_list_is_offline_and_does_not_create_output(self):
        with patch.object(downloader, "download_file", side_effect=AssertionError("network")):
            self.assertEqual(self.run_main("--list"), 0)
        self.assertFalse((self.root / "weights").exists())

    def test_check_missing_is_nonzero_without_download_or_writes(self):
        with patch.object(downloader, "download_file", side_effect=AssertionError("network")):
            self.assertEqual(self.run_main("--check"), 1)
        self.assertFalse((self.root / "weights").exists())

    def test_existing_correct_file_is_reused(self):
        path = self.root / "weights/moge/model.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(self.data)
        with patch.object(downloader, "download_file", side_effect=AssertionError("network")):
            self.assertEqual(self.run_main(), 0)

    def test_existing_bad_file_is_not_overwritten(self):
        path = self.root / "weights/moge/model.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"unrelated file")
        with patch.object(downloader, "download_file", side_effect=AssertionError("network")):
            self.assertEqual(self.run_main(), 1)
        self.assertEqual(path.read_bytes(), b"unrelated file")

    def test_direct_download_is_verified_before_publication(self):
        path = self.root / "raft/model.pt"
        with patch.object(downloader, "urlopen", return_value=io.BytesIO(self.data)):
            downloader.download_file(self.model, self.entry, path)
        self.assertEqual(path.read_bytes(), self.data)
        self.assertEqual(list(path.parent.glob("*.partial")), [])

    def test_partial_or_html_response_is_not_published(self):
        path = self.root / "raft/model.pt"
        with self.assertRaises(ValueError):
            downloader.publish_stream(io.BytesIO(b"<html>error</html>"), path, self.entry)
        self.assertFalse(path.exists())
        self.assertEqual(list(path.parent.iterdir()), [])

    def test_concurrent_destination_is_preserved(self):
        path = self.root / "model.pt"
        path.write_bytes(b"another writer")
        with self.assertRaises(FileExistsError):
            downloader.publish_stream(io.BytesIO(self.data), path, self.entry)
        self.assertEqual(path.read_bytes(), b"another writer")

    def test_cached_huggingface_file_is_copied_without_network(self):
        cached = self.root / "cached.pt"
        cached.write_bytes(self.data)
        model = {**self.model, "repo_id": "public/model", "revision": "a" * 40}
        with patch("huggingface_hub.try_to_load_from_cache", return_value=str(cached)), \
             patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("network")):
            path = self.root / "moge/model.pt"
            downloader.download_file(model, self.entry, path)
        self.assertEqual(path.read_bytes(), self.data)
        self.assertFalse(path.is_symlink())

    def test_hf_nested_files_use_correct_root_and_pinned_revision(self):
        entry = entry_for(self.data, "nested/tokenizer.json")
        model = {**self.model, "repo_id": "public/model", "revision": "b" * 40}
        path = self.root / "wan/nested/tokenizer.json"

        def fake_download(**kwargs):
            self.assertEqual(kwargs["local_dir"], self.root / "wan/.download-staging")
            self.assertEqual(kwargs["filename"], entry[0])
            self.assertEqual(kwargs["revision"], "b" * 40)
            staged = kwargs["local_dir"] / entry[0]
            staged.parent.mkdir(parents=True)
            staged.write_bytes(self.data)
            return str(staged)

        with patch("huggingface_hub.try_to_load_from_cache", return_value=None), \
             patch("huggingface_hub.hf_hub_download", side_effect=fake_download):
            downloader.download_file(model, entry, path)
        self.assertEqual(path.read_bytes(), self.data)
        self.assertFalse((self.root / "wan/.download-staging/nested/tokenizer.json").exists())

    def test_corrupt_hf_download_never_reaches_runtime_path(self):
        model = {**self.model, "repo_id": "public/model", "revision": "b" * 40}
        path = self.root / "moge/model.pt"
        staged = self.root / "moge/.download-staging/model.pt"
        staged.parent.mkdir(parents=True)
        staged.write_bytes(b"x" * len(self.data))
        with patch("huggingface_hub.try_to_load_from_cache", return_value=None), \
             patch("huggingface_hub.hf_hub_download", return_value=str(staged)), \
             self.assertRaises(ValueError):
            downloader.download_file(model, self.entry, path)
        self.assertFalse(path.exists())
        self.assertFalse(staged.exists())

    def test_network_error_is_nonzero_and_does_not_echo_signed_url(self):
        with patch.object(downloader, "download_file", side_effect=OSError("secret signed URL")):
            self.assertEqual(self.run_main(), 1)

    def test_reject_manifest_path_traversal(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps({"bad": {**self.model, "directory": "../outside"}}))
        with patch.object(downloader, "MANIFEST_PATH", path), self.assertRaises(ValueError):
            downloader.load_manifest()


if __name__ == "__main__":
    unittest.main()