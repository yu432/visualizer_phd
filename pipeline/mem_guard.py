"""Run a child process while watching system memory, and kill it before the machine freezes.

Two things were learned the hard way on this machine, and both shape this wrapper.

First, there is no OS-level safety net: macOS had no swap configured, so memory pressure
turns into a full system hang instead of an out of memory kill.

Second, and the reason the first version of this file was useless, process RSS is blind to
the real consumer. While a 263-frame VGGT-omega run drove the machine into a freeze, the
child tree reported 8.6 GiB resident while system available memory fell from 26 to 2.1 GiB.
The model stores its activations in Metal buffers, which land in wired pages owned by the
GPU driver, not in the child's resident size. So the primary tripwire here is the system
available figure, sampled often and with a wide margin, and the child RSS cap is only a
secondary check.

Once the system actually starts thrashing the monitor starves and cannot kill anything, so
the margin matters more than the sampling rate: kill while memory is still freely available.
"""
import argparse
import json
import signal
import subprocess
import time
from pathlib import Path

import psutil


def tree_rss(process):
    """RSS of the child and every descendant, in bytes."""
    total = 0
    for part in [process] + process.children(recursive=True):
        try:
            total += part.memory_info().rss
        except psutil.Error:
            continue
    return total


def write_state(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + chr(10))
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-available-gib", type=float, default=10.0,
                        help="kill as soon as system available memory drops below this")
    parser.add_argument("--limit-gib", type=float, default=0.0,
                        help="secondary cap on child tree RSS, 0 disables it")
    parser.add_argument("--require-free-gib", type=float, default=0.0,
                        help="refuse to start unless this much memory is available")
    parser.add_argument("--max-seconds", type=float, default=0.0,
                        help="kill the child after this long, 0 disables the limit")
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--report-every", type=float, default=30.0)
    parser.add_argument("--state", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given")

    state = Path(args.state)
    state.parent.mkdir(parents=True, exist_ok=True)
    virtual = psutil.virtual_memory()
    available = virtual.available / 1024 ** 3
    baseline = dict(available_gib=available, wired_gib=virtual.wired / 1024 ** 3,
                    swap_used_gib=psutil.swap_memory().used / 1024 ** 3,
                    command=command)

    # A killed or finished MPS run does not hand its unified memory back at once: 19-27 GiB of
    # wired pages have been seen while every process together used 6.8 GiB. Without a preflight
    # a follow-up run starts into a starved machine and freezes on the first forward pass.
    if args.require_free_gib and available < args.require_free_gib:
        write_state(state, dict(baseline=baseline, status="no_headroom",
                                elapsed_seconds=0.0, peak_child_rss_gib=0.0,
                                peak_system_gib=0.0, min_free_gib=available,
                                min_available_gib=args.min_available_gib,
                                require_free_gib=args.require_free_gib))
        print("mem_guard: refusing to start, %.1f GiB available but %.1f GiB required"
              % (available, args.require_free_gib), flush=True)
        return 3

    started = time.monotonic()
    peak_rss = 0.0
    min_available = available
    max_swap = baseline["swap_used_gib"]
    status = "completed"
    proc = subprocess.Popen(command, start_new_session=True)
    child = psutil.Process(proc.pid)
    last_report = started

    def signal_tree(sig):
        """Signal every live process in the child tree.

        os.killpg is not reliable here: a child that ignores SIGTERM and then has to be
        SIGKILLed has been seen to answer EPERM, which left the whole run unmonitored.
        Signalling each process id individually is always permitted for our own children.
        """
        for part in [child] + child.children(recursive=True):
            try:
                part.send_signal(sig)
            except psutil.Error:
                continue

    def kill(reason):
        """Kill without a grace period.

        A grace period is not free: while SIGTERM is pending the child keeps allocating.
        A test child ignoring SIGTERM grew from 2 to 6 GiB during a ten second wait, and the
        peak recorded for it was the 2 GiB that triggered the kill. Whatever the run was
        doing is lost either way, so the tree is SIGKILLed at once.
        """
        try:
            signal_tree(signal.SIGKILL)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        except Exception as error:  # never let the kill path mask the real failure
            print("mem_guard: kill raised %r" % (error,), flush=True)
        print("mem_guard: killed child tree (%s) at available %.1f GiB, rss %.1f GiB"
              % (reason, min_available, peak_rss), flush=True)

    try:
        while proc.poll() is None:
            virtual = psutil.virtual_memory()
            current_available = virtual.available / 1024 ** 3
            rss = tree_rss(child) / 1024 ** 3 if args.limit_gib else 0.0
            peak_rss = max(peak_rss, rss)
            min_available = min(min_available, current_available)
            max_swap = max(max_swap, psutil.swap_memory().used / 1024 ** 3)
            if current_available < args.min_available_gib:
                status = "system_pressure"
            elif args.limit_gib and rss > args.limit_gib:
                status = "memory_limited"
            elif args.max_seconds and time.monotonic() - started > args.max_seconds:
                status = "timeout"
            if status != "completed":
                kill(status)
                break
            now = time.monotonic()
            if now - last_report >= args.report_every:
                print("mem_guard: %.0f s available %.1f GiB wired %.1f GiB swap %.1f GiB rss %.1f GiB"
                      % (now - started, current_available, virtual.wired / 1024 ** 3,
                         psutil.swap_memory().used / 1024 ** 3, rss), flush=True)
                last_report = now
                write_state(state, dict(baseline=baseline, status="running", pid=proc.pid,
                                        elapsed_seconds=now - started,
                                        peak_child_rss_gib=peak_rss,
                                        peak_system_gib=available - min_available,
                                        min_free_gib=min_available, max_swap_gib=max_swap,
                                        min_available_gib=args.min_available_gib,
                                        limit_gib=args.limit_gib))
            time.sleep(args.interval)
    except BaseException:
        kill("monitor_failed")
        raise
    finally:
        if proc.poll() is None:
            kill("finally")

    summary = dict(baseline=baseline, status=status, pid=proc.pid, returncode=proc.returncode,
                   elapsed_seconds=time.monotonic() - started,
                   peak_child_rss_gib=peak_rss, peak_system_gib=available - min_available,
                   min_free_gib=min_available, max_swap_gib=max_swap,
                   min_available_gib=args.min_available_gib, limit_gib=args.limit_gib,
                   max_seconds=args.max_seconds)
    write_state(state, summary)
    print("mem_guard: %s after %.0f s, memory used %.1f GiB, lowest available %.1f GiB, "
          "swap peak %.1f GiB, child rc %s"
          % (status, summary["elapsed_seconds"], summary["peak_system_gib"],
             min_available, max_swap, proc.returncode), flush=True)
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
