"""Pure versioned segment seed derivation; continuation retains public aliases."""
import hashlib

from sep_research import protocol
from sep_research.seed_control import frozen_seed_guard

DOMAIN = 'sep_amendment_segment_rng_v1'
# Fixed envelope, independent of family/device/current resource geometry.
MAX_WORKERS = 64


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'invalid {name}')
    return value


def _forbidden():
    return set(frozen_seed_guard()['reserved_values']) | {
        protocol.VALIDATION_ROOT, protocol.TEST_ROOT,
        protocol.DEVELOPMENT_ROOT, protocol.BOOTSTRAP_ROOT}


def _safe_seed(seed, forbidden):
    return (0 <= seed < 2**31 - 1001 * MAX_WORKERS
            and all(seed + offset not in forbidden for w in range(MAX_WORKERS + 1)
                    for offset in (w, 1001 * w)))


def derive_segment_seed(base_training_seed, start_iteration):
    """Versioned SHA256 seed; same base/boundary means same stream for all families.

    Reject reserved/protected base and effective seeds. Collision retries are
    deterministic and independent of attempts, time, device, family and budget.
    Worker globals use seed+w; physical resets use seed+1001*w (one vector).
    """
    base = _integer(base_training_seed, 'base_training_seed')
    start = _integer(start_iteration, 'start_iteration')
    forbidden = _forbidden()
    if not _safe_seed(base, forbidden):
        raise ValueError('reserved/protected/out-of-range base or effective seed')
    counter = 0
    while True:
        payload = f'{DOMAIN}:{base}:{start}:{counter}'.encode('ascii')
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') % (2**31)
        if seed != base and _safe_seed(seed, forbidden):
            return seed
        counter += 1


