"""Small portable JSON/file contracts. Native bundles are trusted operator inputs."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe(path):
    path = Path(os.path.abspath(path))
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('symlink path forbidden')
    return path


def relative(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('invalid portable relative path')
    result = safe(Path(root) / name)
    if not result.is_relative_to(safe(root)):
        raise ValueError('path escapes root')
    return result


def tree(root):
    root = safe(root)
    result = {}
    for p in sorted(root.rglob('*')):
        safe(p)
        if p.is_file():
            result[str(p.relative_to(root))] = sha(p)
    return result


def write(path, value, *, replace=False):
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.publish_', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def evidence_tree(root):
    """Allowlist scientific result files, excluding logs/caches/spill/credentials."""
    root = safe(root)
    if (root / 'interrupted.json').exists():
        from .interrupted import evidence_tree as interrupted_tree
        return interrupted_tree(root)
    if (root / 'partial_recovery.json').exists():
        from .recovery_partial import evidence_tree as partial_recovery_tree
        return partial_recovery_tree(root)
    if (root / 'partial.json').exists():
        from .partial import evidence_tree as partial_tree
        return partial_tree(root)
    result = read(root / 'result.json')
    names = ['completion.json', 'result.json', 'plan.json', 'preflight.json',
             'writer.json', 'metrics.jsonl', 'seed_audit.json', 'resources.jsonl']
    files = {name: sha(relative(root, name)) for name in names}
    for entry in result['checkpoints']:
        bundle = relative(root, entry['path'])
        files.update({entry['path']+'/'+name: value for name, value in tree(bundle).items()})
    if result['source']:
        files.update({'source/'+name: value for name, value in evidence_tree(root / 'source').items()})
    return files


def copy_evidence(source, destination):
    import shutil
    files = evidence_tree(source)
    destination = safe(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name in files:
        target = relative(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(relative(source, name), target)
    if evidence_tree(destination) != files:
        raise ValueError('portable scientific copy mismatch')
    return files
