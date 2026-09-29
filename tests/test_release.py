"""Isolated stdlib checks; never seal or import the scientific package."""

import contextlib
import gzip
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True

SCRIPT = Path(__file__).absolute().parents[1] / "scripts" / "release.py"
spec = importlib.util.spec_from_file_location("release", SCRIPT)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "source"
        self.root.mkdir()
        self.names = [release.ALLOWLIST, "code/scientificmodels.py", "data/sample.json.gz",
                      "data/table.csv", "data/array.npz", "results/figure.pdf"]
        for name, data in {
            "code/scientificmodels.py": b'# secret/password and /workspace are ordinary documentation\n',
            "data/sample.json.gz": gzip.compress(b'{"value": 1}'),
            "data/table.csv": b"x,y\n1,2\n", "data/array.npz": b"binary fixture",
            "results/figure.pdf": b"%PDF fixture",
        }.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.write_list()

    def write_list(self):
        (self.root / release.ALLOWLIST).write_text("\n".join(self.names) + "\n", encoding="utf-8")

    def run_cli(self, *args, script=SCRIPT):
        return subprocess.run([sys.executable, "-I", str(script), *map(str, args)],
                              cwd=self.base, capture_output=True, text=True)

    def test_seal_verify_export_and_exported_default_root(self):
        copied_script = self.root / "scripts/release.py"
        copied_script.parent.mkdir()
        shutil.copyfile(SCRIPT, copied_script)
        self.names.append("scripts/release.py")
        self.write_list()
        for args in [("--root", self.root, "seal"), ("verify", "--root", self.root)]:
            result = self.run_cli(*args)
            self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((self.root / release.MANIFEST).read_text())
        self.assertEqual(set(manifest["files"]), set(self.names))
        self.assertEqual(manifest["total_bytes"], sum((self.root / n).stat().st_size for n in self.names))
        destination = self.base / "export"
        result = self.run_cli("export", "--root", self.root, "--destination", destination)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()},
                         set(self.names) | {release.MANIFEST})
        for name in self.names + [release.MANIFEST]:
            self.assertEqual((self.root / name).read_bytes(), (destination / name).read_bytes())
        result = self.run_cli("verify", script=destination / "scripts/release.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        release.seal(self.root)
        self.assertEqual(manifest, json.loads((self.root / release.MANIFEST).read_text()))

    def test_existing_destination_preserves_bytes(self):
        release.seal(self.root)
        for kind in ("directory", "file", "symlink"):
            with self.subTest(kind=kind):
                target = self.base / kind
                if kind == "directory":
                    target.mkdir()
                    sentinel = target / "keep"
                    sentinel.write_bytes(b"unchanged")
                elif kind == "file":
                    target.write_bytes(b"unchanged")
                    sentinel = target
                else:
                    target.symlink_to(self.base / "missing")
                    sentinel = None
                with self.assertRaisesRegex(release.ReleaseError, "already exists"):
                    release.export(self.root, target)
                if sentinel:
                    self.assertEqual(sentinel.read_bytes(), b"unchanged")
                else:
                    self.assertTrue(target.is_symlink())

    def test_tampered_file_blocks_verify_and_export(self):
        release.seal(self.root)
        (self.root / "data/table.csv").write_bytes(b"tampered")
        with self.assertRaises(release.ReleaseError):
            release.verified_snapshot(self.root)
        destination = self.base / "export"
        with self.assertRaises(release.ReleaseError):
            release.export(self.root, destination)
        self.assertFalse(destination.exists())

    def test_changed_list_and_manifest_metadata(self):
        release.seal(self.root)
        manifest_path = self.root / release.MANIFEST
        original = manifest_path.read_bytes()
        for field, value in [("total_bytes", 0), ("version", True), ("files", {})]:
            with self.subTest(field=field):
                manifest = json.loads(original)
                manifest[field] = value
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaises(release.ReleaseError):
                    release.verified_snapshot(self.root)
        manifest_path.write_bytes(original)
        self.names.remove("data/table.csv")
        self.write_list()
        with self.assertRaises(release.ReleaseError):
            release.verified_snapshot(self.root)

    def test_duplicate_manifest_keys(self):
        release.seal(self.root)
        path = self.root / release.MANIFEST
        path.write_text(path.read_text().replace('"version": 1', '"version": 1, "version": 1'))
        with self.assertRaisesRegex(release.ReleaseError, "duplicate keys"):
            release.verified_snapshot(self.root)

    def test_invalid_paths_and_banned_categories(self):
        bad_names = ["../escape", "/absolute", "a/../b", "a//b", "./a", "a\\b", "C:/a", "",
                     ".git/config", ".env", ".env.production", "keys/id_rsa", "keys/server.pem",
                     "_local/notes", "private/notes", "author/notes", "build/x", "__pycache__/x",
                     ".cache/x", "checkpoint/x", "model.pkl", "model.pickle", "model.pt",
                     "model.pth", "model.ckpt", "pkg.whl", "model.pkl.gz", "keys/id_ecdsa", ".ssh/config", "articles/paper.pdf"]
        for name in bad_names:
            with self.subTest(name=name):
                with self.assertRaises(release.ReleaseError):
                    release.checked_path(self.root, name)

    def test_allowlist_contract(self):
        original = self.names[:]
        for names in [original + [original[0]], original[1:], original + [release.MANIFEST],
                      original + ["../escape"], original + [""]]:
            self.names = names
            self.write_list()
            with self.assertRaises(release.ReleaseError):
                release.seal(self.root)

    def test_symlinks_and_nonregular_files(self):
        (self.root / "link").symlink_to(self.root / "data/table.csv")
        (self.root / "linked-dir").symlink_to(self.root / "data", target_is_directory=True)
        for name in ["link", "linked-dir/table.csv", "data"]:
            with self.subTest(name=name):
                with self.assertRaises(release.ReleaseError):
                    release.checked_path(self.root, name)
        alias = self.base / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(release.ReleaseError):
            release.snapshot(alias)
        (self.root / release.MANIFEST).symlink_to(self.base / "missing")
        with self.assertRaises(release.ReleaseError):
            release.seal(self.root)
        self.assertFalse((self.base / "missing").exists())

    def test_size_bounds(self):
        big = self.root / "big.txt"
        with big.open("wb") as stream:
            stream.truncate(release.MAX_FILE + 1)
        with self.assertRaisesRegex(release.ReleaseError, "file limit"):
            release.read_file(self.root, "big.txt")
        total = sum((self.root / name).stat().st_size for name in self.names)
        with patch.object(release, "MAX_TOTAL", total):
            release.snapshot(self.root)
        with patch.object(release, "MAX_TOTAL", total - 1):
            with self.assertRaisesRegex(release.ReleaseError, "total limit"):
                release.snapshot(self.root)
        bomb = gzip.compress(b"x" * (release.MAX_TEXT + 1))
        self.assertLess(len(bomb), release.MAX_FILE)
        with self.assertRaisesRegex(release.ReleaseError, "decompressed text exceeds 50 MiB limit"):
            release.check_content("data/bomb.json.gz", bomb)

    def test_compressed_text_above_file_limit_is_accepted(self):
        text = b'{"value": "' + b"x" * release.MAX_FILE + b'"}'
        self.assertGreater(len(text), release.MAX_FILE)
        self.assertLess(len(text), release.MAX_TEXT)
        data = gzip.compress(text)
        self.assertLess(len(data), release.MAX_FILE)
        name = "data/sample.json.gz"
        (self.root / name).write_bytes(data)
        release.check_content(name, release.read_file(self.root, name))

    def test_jsonl_gzip_training_input(self):
        release.check_content("data/history.jsonl.gz", gzip.compress(b'{"x": 1}\n{"x": 2}\n'))

    def test_obvious_secrets_do_not_leak_values(self):
        secrets = [b"-----BEGIN " + b"PRIVATE KEY-----", b"ghp_" + b"a" * 36,
                   b"github_pat_" + b"a" * 30, b"xoxb-" + b"1234567890123456",
                   b"AKIA" + b"ABCDEFGHIJKLMNOP", b"sk-proj-" + b"a" * 40]
        name = "data/credentials.csv.gz"
        self.names.append(name)
        self.write_list()
        for secret in secrets:
            with self.subTest(prefix=secret[:4]):
                (self.root / name).write_bytes(gzip.compress(secret))
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    status = release.main(["seal", "--root", str(self.root)])
                self.assertEqual(status, 1)
                self.assertIn(name, stderr.getvalue())
                self.assertNotIn(secret.decode(), stderr.getvalue())
                with self.assertRaises(release.ReleaseError):
                    release.check_content("code/example.py", secret)

    def test_unlisted_private_files_never_read_or_copied(self):
        sensitive = self.root / "root.sensitive"
        sensitive.write_bytes(b"-----BEGIN " + b"PRIVATE KEY-----")
        (self.root / "_local").mkdir()
        (self.root / "_local/private").symlink_to(self.base / "missing")
        original_read = release.read_file
        reads = []

        def tracked(root, name):
            reads.append(name)
            self.assertIn(name, self.names + [release.MANIFEST])
            return original_read(root, name)

        with patch.object(release, "read_file", side_effect=tracked):
            release.seal(self.root)
            release.verified_snapshot(self.root)
            release.export(self.root, self.base / "export")
        self.assertTrue(reads)
        self.assertFalse((self.base / "export/root.sensitive").exists())
        self.assertTrue(sensitive.exists())

    def test_partial_export_reports_without_cleanup(self):
        release.seal(self.root)
        destination = self.base / "partial"
        original_open = Path.open

        def fail_copy(path, *args, **kwargs):
            if path == destination / "code/scientificmodels.py":
                raise OSError("simulated disk failure")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", fail_copy):
            with self.assertRaisesRegex(release.ReleaseError, "partial files; no cleanup"):
                release.export(self.root, destination)
        self.assertTrue((destination / release.ALLOWLIST).is_file())


if __name__ == "__main__":
    unittest.main()
