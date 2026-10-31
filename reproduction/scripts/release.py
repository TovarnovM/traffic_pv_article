"""Seal, verify, and copy the exact public release allowlist using only stdlib.

Content checks catch obvious credentials, not every possible publication secret.
Unlisted files are never inspected. Run against a quiescent source tree.
"""

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zlib

ALLOWLIST = "RELEASE_FILES.txt"
MANIFEST = "MANIFEST.json"
MAX_FILE = 10 * 1024 * 1024
MAX_TOTAL = 50 * 1024 * 1024
MAX_TEXT = 50 * 1024 * 1024
BANNED_PARTS = {
    ".git", ".env", "_local", "private", "author", "private_author",
    "author_private", "_private", ".ssh", ".aws", "cache", "caches", "build", "dist", "__pycache__", ".cache", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".venv", "venv", "node_modules",
    "checkpoint", "checkpoints", "wheels",
}
BANNED_SUFFIXES = {
    ".pem", ".key", ".p12", ".pfx", ".pkl", ".pickle", ".pt", ".pth",
    ".ckpt", ".whl", ".pyc", ".pyo",
}
SECRET_PATTERNS = (
    ("private key", re.compile(rb"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")),
    ("credential token", re.compile(
        rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
        rb"|xox[baprs]-[A-Za-z0-9-]{10,}|AKIA[A-Z0-9]{16}"
        rb"|sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,})\b"
    )),
)


class ReleaseError(Exception):
    pass


def checked_path(root, name):
    """Validate spelling and every ancestor without resolving away symlinks."""
    if (not name or "\\" in name or ":" in name
            or any(ord(c) < 32 or ord(c) == 127 for c in name)):
        raise ReleaseError("invalid release path")
    parts = name.split("/")
    if PurePosixPath(name).is_absolute() or any(p in {"", ".", ".."} for p in parts):
        raise ReleaseError("invalid release path")
    for part in parts:
        lower = part.lower()
        if (lower in BANNED_PARTS or lower.startswith(".env.")
                or lower.startswith(("id_rsa", "id_ed25519", "id_dsa", "id_ecdsa"))
                or any(suffix in BANNED_SUFFIXES for suffix in Path(lower).suffixes)):
            raise ReleaseError(f"{name}: prohibited file category")
    if Path(name).suffix.lower() == ".pdf" and parts[0] != "results":
        raise ReleaseError(f"{name}: PDF outside results")
    path = root / name
    for ancestor in reversed((path, *path.parents)):
        if ancestor.is_symlink():
            raise ReleaseError(f"{name}: symlink prohibited")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ReleaseError(f"{name}: not a regular file")
    return path


def read_file(root, name):
    path = checked_path(root, name)
    # The selected-policy archive is the only explicitly allowed large file.
    limit = 40 * 1024 * 1024 if name == "weights/selected_policies.zip" else MAX_FILE
    if path.stat().st_size > limit:
        raise ReleaseError(f"{name}: exceeds file size limit")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ReleaseError(f"{name}: exceeds file size limit")
    return data


def check_content(name, data):
    lower = name.lower()
    if lower.endswith((".pdf", ".png", ".jpg", ".jpeg", ".gif", ".npz", ".npy")):
        return
    if lower.endswith(".gz"):
        if not lower.endswith((".json.gz", ".jsonl.gz", ".csv.gz")):
            raise ReleaseError(f"{name}: unsupported compressed file")
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
                data = stream.read(MAX_TEXT + 1)
        except (OSError, EOFError, zlib.error):
            raise ReleaseError(f"{name}: invalid gzip data") from None
        if len(data) > MAX_TEXT:
            raise ReleaseError(f"{name}: decompressed text exceeds 50 MiB limit")
    for category, pattern in SECRET_PATTERNS:
        if pattern.search(data):
            raise ReleaseError(f"{name}: {category} detected")


def snapshot(root):
    allowlist = read_file(root, ALLOWLIST)
    try:
        names = allowlist.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        raise ReleaseError("RELEASE_FILES.txt: expected UTF-8") from None
    if len(names) != len(set(names)):
        raise ReleaseError("RELEASE_FILES.txt: duplicate paths")
    if ALLOWLIST not in names or MANIFEST in names:
        raise ReleaseError("allowlist must include RELEASE_FILES.txt and exclude MANIFEST.json")
    files = {}
    total = 0
    for name in names:
        data = allowlist if name == ALLOWLIST else read_file(root, name)
        check_content(name, data)
        total += len(data)
        if total > MAX_TOTAL:
            raise ReleaseError("release exceeds 50 MiB total limit")
        files[name] = data
    if "weights/selected_policies.zip" in files:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from policies import verify_weights
        try:
            verify_weights(root / "weights")
        except ValueError as exc:
            raise ReleaseError(str(exc)) from exc
    manifest = {
        "version": 1,
        "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                  for name, data in sorted(files.items())},
        "total_bytes": total,
    }
    return files, manifest


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseError("MANIFEST.json: duplicate keys")
        result[key] = value
    return result


def verified_snapshot(root):
    files, expected = snapshot(root)
    raw = read_file(root, MANIFEST)
    try:
        actual = json.loads(raw, object_pairs_hook=unique_object)
    except (ValueError, UnicodeDecodeError):
        raise ReleaseError("MANIFEST.json: invalid JSON") from None
    # Canonical JSON comparison also distinguishes booleans/floats from integers.
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ReleaseError("MANIFEST.json: file list, hashes, sizes, or metadata mismatch")
    return files, raw


def seal(root):
    _, manifest = snapshot(root)
    target = root / MANIFEST
    if os.path.lexists(target):
        checked_path(root, MANIFEST)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def export(root, destination):
    if os.path.lexists(destination):
        raise ReleaseError("destination already exists; refusing export")
    files, manifest = verified_snapshot(root)
    # Exclusive creation preserves any existing destination, including dangling links.
    destination.mkdir()
    try:
        for name, data in {**files, MANIFEST: manifest}.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
    except OSError:
        raise ReleaseError("export failed; new destination may contain partial files; no cleanup performed") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).absolute().parents[1])
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("seal", "verify", "export"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--root", type=Path, default=argparse.SUPPRESS)
        if command == "export":
            subparser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(os.path.abspath(args.root))
    try:
        if args.command == "seal":
            seal(root)
        elif args.command == "verify":
            verified_snapshot(root)
        else:
            export(root, args.destination)
    except ReleaseError as exc:
        print(f"release: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        # Do not echo file contents or arbitrary OS exception payloads.
        print(f"release: filesystem operation failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    print(f"release: {args.command} successful")
    return 0


if __name__ == "__main__":
    sys.exit(main())
