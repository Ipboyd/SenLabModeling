# profile_runner.py
import argparse, os, time, subprocess, psutil, shutil, sys
from pathlib import Path

def find_exe(dirpath: Path):
    # Brian names the target 'main' by default
    for name in ('main.exe', 'main'):
        p = dirpath / name
        if p.exists():
            return p
    # Fallback: first .exe (Windows) or any file without extension (Unix)
    exe = next((p for p in dirpath.glob('*.exe')), None)
    if exe: return exe
    for p in dirpath.iterdir():
        if p.is_file() and os.access(p, os.X_OK):
            return p
    raise FileNotFoundError(f"No executable found in {dirpath}")

def run_and_profile(exe: Path, omp_threads: int) -> tuple[float, float]:
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(omp_threads)
    t0 = time.perf_counter()
    # Start process
    proc = psutil.Popen([str(exe)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak = 0
    try:
        while proc.is_running():
            try:
                mem = proc.memory_info().rss
                peak = max(peak, mem)
            except psutil.Error:
                pass
            time.sleep(0.05)
        rc = proc.wait()
    finally:
        try:
            proc.kill()
        except Exception:
            pass
    dt = time.perf_counter() - t0
    return dt, peak / (1024**2)  # seconds, MB

def run_many_and_profile(exe: Path, num_procs: int, omp_threads_each: int) -> tuple[float, float]:
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(omp_threads_each)
    procs = [psutil.Popen([str(exe)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             for _ in range(num_procs)]
    t0 = time.perf_counter()
    peak_sum = 0
    try:
        # Track per-proc peaks
        peaks = {p.pid: 0 for p in procs}
        while any(p.is_running() for p in procs):
            for p in list(procs):
                try:
                    mem = p.memory_info().rss
                    peaks[p.pid] = max(peaks[p.pid], mem)
                except psutil.Error:
                    pass
            time.sleep(0.05)
        for p in procs:
            p.wait()
        peak_sum = sum(peaks.values()) / (1024**2)
    finally:
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass
    dt = time.perf_counter() - t0
    return dt, peak_sum  # seconds, MB

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairwise_dir", default="cpp_pairwise_10x10", help="Folder with pairwise build")
    ap.add_argument("--single_dir",   default="cpp_single_1neuron", help="Folder with single-neuron build")
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 10], help="Thread counts to test for pairwise")
    ap.add_argument("--single_procs", type=int, default=10, help="#processes for single-neuron parallel run")
    ap.add_argument("--single_threads_each", type=int, default=1, help="OMP threads per single-neuron proc")
    args = ap.parse_args()

    pairwise_exe = find_exe(Path(args.pairwise_dir))
    single_exe   = find_exe(Path(args.single_dir))

    print(f"[Pairwise] EXE: {pairwise_exe}")
    print(f"[Single]   EXE: {single_exe}")

    print("\n=== Pairwise 10x10 (single process; varying OMP threads) ===")
    for th in args.threads:
        sec, mb = run_and_profile(pairwise_exe, th)
        print(f"OMP_NUM_THREADS={th:>2}  time={sec:7.3f}s  peakRSS={mb:7.1f} MB")

    print("\n=== Single 1-neuron (N parallel processes) ===")
    sec, mb = run_many_and_profile(single_exe, args.single_procs, args.single_threads_each)
    print(f"Nprocs={args.single_procs}  each_threads={args.single_threads_each}  time={sec:7.3f}s  totalPeakRSS={mb:7.1f} MB")

if __name__ == "__main__":
    try:
        import psutil  # noqa
    except ImportError:
        print("Please: pip install psutil")
        sys.exit(1)
    main()

