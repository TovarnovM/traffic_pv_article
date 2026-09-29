"""Container cgroup accounting; host/driver measurements are supplementary."""
import os
from pathlib import Path
import shutil
import threading
import time

from .common import write


def _number(path):
    value = path.read_text().strip()
    return None if value == 'max' else int(value)


def snapshot(output, cgroup=Path('/sys/fs/cgroup')):
    import psutil
    if (cgroup / 'memory.current').exists():
        events = dict(line.split() for line in (cgroup / 'memory.events').read_text().splitlines())
        quota, period = (cgroup / 'cpu.max').read_text().split()
        memory = dict(version=2, current_bytes=_number(cgroup / 'memory.current'),
            peak_bytes=_number(cgroup / 'memory.peak'), limit_bytes=_number(cgroup / 'memory.max'),
            swap_current_bytes=_number(cgroup / 'memory.swap.current'),
            swap_limit_bytes=_number(cgroup / 'memory.swap.max'),
            events={k: int(v) for k, v in events.items()},
            cpu_quota_logical=None if quota == 'max' else int(quota) / int(period),
            cpu_stat=(cgroup / 'cpu.stat').read_text())
    else:
        cg = cgroup / 'memory'
        cpu = cgroup / 'cpu'
        quota = _number(cpu / 'cpu.cfs_quota_us')
        limit = _number(cg / 'memory.limit_in_bytes')
        memsw = _number(cg / 'memory.memsw.limit_in_bytes')
        memory = dict(version=1, current_bytes=_number(cg / 'memory.usage_in_bytes'),
            peak_bytes=_number(cg / 'memory.max_usage_in_bytes'), limit_bytes=limit,
            swap_current_bytes=_number(cg / 'memory.memsw.usage_in_bytes') - _number(cg / 'memory.usage_in_bytes'),
            swap_limit_bytes=max(0, memsw-limit), events={'failcnt': _number(cg / 'memory.failcnt')},
            cpu_quota_logical=None if quota < 0 else quota / _number(cpu / 'cpu.cfs_period_us'),
            cpu_stat=(cpu / 'cpu.stat').read_text())
    process = psutil.Process()
    processes = []
    for proc in [process, *process.children(recursive=True)]:
        try:
            processes.append(dict(pid=proc.pid, threads=proc.num_threads(),
                cpu_seconds=sum(proc.cpu_times()[:2]), pss_bytes=proc.memory_full_info().pss))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    shm = shutil.disk_usage('/dev/shm')
    return dict(time_unix=time.time(), cgroup=memory,
        host_mem_available_bytes=psutil.virtual_memory().available,
        affinity_logical_cpus=len(os.sched_getaffinity(0)), processes=processes,
        disk_free_bytes=shutil.disk_usage(output).free,
        shm_capacity_bytes=shm.total, shm_used_bytes=shm.used)


def check_limits(sample, profile):
    cg = sample['cgroup']
    # Linux rounds a byte-denominated Docker memory limit down to whole pages.
    # Accept the requested value (test/v1 accounting) or its exact kernel rounding,
    # never arbitrary lower limits and never more memory than was requested.
    requested = profile['memory_limit_bytes']
    rounded = requested // os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PAGE_SIZE')
    if (cg['limit_bytes'] not in (requested, rounded) or cg['swap_limit_bytes'] != 0
            or cg['cpu_quota_logical'] != profile['cpu_quota_logical']
            or sample['shm_capacity_bytes'] < profile['shm_bytes']
            or sample['affinity_logical_cpus'] < profile['cpu_quota_logical']
            or sample['disk_free_bytes'] < profile['disk_min_bytes']):
        raise ValueError('actual cgroup CPU/memory/swap/shm/disk limits do not match CPU profile')
    if any(cg['events'].get(k, 0) for k in ('oom', 'oom_kill', 'failcnt')):
        raise ValueError('cgroup already has OOM/failure events')


class Monitor:
    def __init__(self, output, profile, status):
        self.output, self.profile, self.status = output, profile, status
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import json
        try:
            with (self.output / 'resources.jsonl').open('x') as stream:
                while True:
                    sample = snapshot(self.output)
                    stream.write(json.dumps(sample, sort_keys=True) + '\n')
                    stream.flush()
                    compact = {k:self.status.get(k) for k in ('status','start_iteration','final_iteration','last_checkpoint')}
                    compact.update(run_id=self.status['row']['run_id'],
                                   segment_boundaries=self.status['row']['segment_boundaries'],
                                   time_unix=sample['time_unix'])
                    write(self.output / 'heartbeat.json', compact, replace=True)
                    if self.stop_event.wait(self.profile['resource_interval_seconds']):
                        break
        except BaseException as exc:
            self.error = exc

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=60)
        if self.thread.is_alive():
            raise RuntimeError('resource monitor did not stop')
        if self.error:
            raise self.error
