import time
import os
from pathlib import Path
import muller
from datetime import datetime


def column_name(i: int) -> str:
    return f"column_{i}"


class IOMonitor:

    def __init__(self):
        self.reset()
        self._original_open = None
        self._monitoring = False

    def reset(self):
        self.files_opened = set()

    def start_monitoring(self):
        if self._monitoring:
            return

        self._monitoring = True
        self._original_open = open

        monitor = self

        class MonitoredFile:
            def __init__(self, file_obj, mode, path):
                self.file_obj = file_obj
                self.mode = mode
                self.path = os.path.abspath(str(path))

                path_parts = Path(self.path).parts
                if len(path_parts) < 2 or path_parts[-2] != 'chunks':
                    monitor.files_opened.add(self.path)

            def read(self, size=-1):
                return self.file_obj.read(size)

            def write(self, data):
                return self.file_obj.write(data)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return self.file_obj.__exit__(*args)

            def __getattr__(self, name):
                return getattr(self.file_obj, name)

        def monitored_open(file, mode='r', *args, **kwargs):
            file_obj = monitor._original_open(file, mode, *args, **kwargs)
            return MonitoredFile(file_obj, mode, file)

        import builtins
        builtins.open = monitored_open

    def stop_monitoring(self):
        if not self._monitoring:
            return

        import builtins
        builtins.open = self._original_open
        self._monitoring = False

    def get_stats(self):
        return {
            'files_count': len(self.files_opened),
        }

    def print_files(self, log_file=None):
        if not self.files_opened:
            msg = "  No files accessed."
            print(msg)
            if log_file:
                log_file.write(msg + "\n")
            return

        msg = f"\n  Files Accessed ({len(self.files_opened)} files):"
        print(msg)
        if log_file:
            log_file.write(msg + "\n")

        for filepath in sorted(self.files_opened):
            try:
                rel_path = os.path.relpath(filepath)
            except:
                rel_path = filepath
            msg = f"    - {rel_path}"
            print(msg)
            if log_file:
                log_file.write(msg + "\n")


class VersionControlBenchmark:
    def __init__(self, log_dir="./logs"):
        self.results = {
            'MULLER': {}
        }
        self.io_stats = {
            'MULLER': {}
        }
        self.data_path = ".output/benchmark_vc"
        self.io_monitor = IOMonitor()

        # Setup log file
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"version_control_{timestamp}.log"
        self.log_path = os.path.join(log_dir, log_filename)
        self.log_file = None

    def open_log(self):
        self.log_file = open(self.log_path, 'w', encoding='utf-8')

    def close_log(self):
        if self.log_file:
            self.log_file.close()

    def log_and_print(self, message):
        print(message)
        if self.log_file:
            self.log_file.write(message + "\n")
            self.log_file.flush()

    def time_function(self, func, monitor_io=True, operation_name="", print_files=False):
        if monitor_io:
            self.io_monitor.reset()
            self.io_monitor.start_monitoring()

        start = time.perf_counter()
        result = func()
        end = time.perf_counter()

        if monitor_io:
            self.io_monitor.stop_monitoring()
            io_stats = self.io_monitor.get_stats()
            if print_files:
                self.io_monitor.print_files(self.log_file)
            return (end - start) * 1000, result, io_stats
        else:
            return (end - start) * 1000, result, None

    # ==================== MULLER ====================
    def setup_muller_dataset(self):
        self.log_and_print("\n[MULLER] Setting up muller dataset...")

        ds = muller.dataset(self.data_path, overwrite=True)
        num_rows = 10000

        for i in range(1):
            ds.create_tensor(column_name(i), htype='generic', dtype='float64')
            getattr(ds, column_name(i)).extend([42.0] * num_rows)

        ds.commit("First commit on main")

        for i in range(1):
            getattr(ds, column_name(i)).extend([42.0] * 1)
        ds.commit("Second commit on main")

        for i in range(1):
            getattr(ds, column_name(i)).extend([42.0] * 1)
        ds.commit("Third commit on main")

        return ds

    def test_muller_view_versions(self, ds):
        def view_versions():
            commits = ds.commits()
            return len(commits)

        elapsed, count, io_stats = self.time_function(view_versions, True, "View Versions", print_files=True)
        self.log_and_print(f"[MULLER] View all versions: {elapsed:.3f} ms (found {count} versions)")
        self.results['MULLER']['view_versions'] = elapsed
        self.io_stats['MULLER']['view_versions'] = io_stats

    def test_muller_create_branch(self, ds):
        def create_branch():
            ds.checkout("test_branch", create=True)
            return True

        elapsed, _, io_stats = self.time_function(create_branch, True, "Create Branch", print_files=True)
        self.log_and_print(f"[MULLER] Create branch: {elapsed:.3f} ms")
        self.results['MULLER']['create_branch'] = elapsed
        self.io_stats['MULLER']['create_branch'] = io_stats

    def test_muller_checkout(self, ds):

        def checkout_branch():
            ds.checkout("main", create=False)
            return True

        elapsed, _, io_stats = self.time_function(checkout_branch, True, "Checkout Branch", print_files=True)
        self.log_and_print(f"[MULLER] Checkout branch: {elapsed:.3f} ms")
        self.results['MULLER']['checkout_branch'] = elapsed
        self.io_stats['MULLER']['checkout_branch'] = io_stats

    def test_muller_delete_branch(self, ds):

        def delete_branch():
            ds.delete_branch("test_branch")
            return True

        elapsed, _, io_stats = self.time_function(delete_branch, True, "Delete Branch", print_files=True)
        self.log_and_print(f"[MULLER] Delete branch: {elapsed:.3f} ms")
        self.results['MULLER']['delete_branch'] = elapsed
        self.io_stats['MULLER']['delete_branch'] = io_stats

    def test_muller_diff(self, ds):
        third_main = ds.commit_id
        ds.checkout("test_1", create=True)
        ds.checkout(third_main, create=False)
        ds.checkout("test_2", create=True)
        for i in range(1):
            getattr(ds, column_name(i)).extend([24.0] * 1)
        fir_test2 = ds.commit("First commit in test_2")

        def diff():
            res = ds.diff(fir_test2, third_main, as_dict=True)
            return res

        elapsed, res, io_stats = self.time_function(diff, True, "Diff", print_files=True)
        self.log_and_print(f"[MULLER] Diff between versions: {elapsed:.3f} ms")
        self.results['MULLER']['diff'] = elapsed
        self.io_stats['MULLER']['diff'] = io_stats

    def test_muller_ffmerge(self, ds):
        ds.checkout("main", create=False)

        def merge_branch():
            ds.merge("test_2")
            return ds

        self.log_and_print(f"[MULLER] Fast-forward merge: before ff-merge length: {len(ds)}")
        elapsed, dataset, io_stats = self.time_function(merge_branch, True, "Fast-Forward Merge", print_files=True)
        self.log_and_print(f"[MULLER] Fast-forward merge: {elapsed:.3f} ms (after ff-merge length: {len(dataset)})")
        self.results['MULLER']['ff_merge'] = elapsed
        self.io_stats['MULLER']['ff_merge'] = io_stats

    def test_detect_conflict(self, ds):
        ds.checkout("test_1", create=False)
        ds.pop()
        fir_1 = ds.commit("First commit in test_1")
        ds.checkout("main", create=False)

        def detect_conf():
            _, conf_res = ds.detect_merge_conflict(target_id=fir_1)
            return conf_res

        elapsed, conf_res, io_stats = self.time_function(detect_conf, True, "Detect Conflict", print_files=True)
        self.log_and_print(f"[MULLER] Detect merge conflict: {elapsed:.3f} ms (conflicts found: {conf_res})")
        self.results['MULLER']['detect_conflict'] = elapsed
        self.io_stats['MULLER']['detect_conflict'] = io_stats

    def test_3_way_merge(self, ds):
        self.log_and_print(f"[MULLER] 3-way merge: before 3-way-merge length: {len(ds)}")

        def three_way_merge():
            ds.merge("test_1", append_resolution="ours", pop_resolution="theirs")
            return ds

        elapsed, ds, io_stats = self.time_function(three_way_merge, True, "3-Way Merge", print_files=True)
        self.log_and_print(f"[MULLER] 3-way merge: {elapsed:.3f} ms (after 3-way merge length: {len(ds)})")
        self.results['MULLER']['3_way_merge'] = elapsed
        self.io_stats['MULLER']['3_way_merge'] = io_stats

    def run_muller_tests(self):
        if muller is None:
            self.log_and_print("\n[MULLER] Skipped - library not available")
            return

        try:
            ds = self.setup_muller_dataset()
            self.test_muller_view_versions(ds)
            self.test_muller_create_branch(ds)
            self.test_muller_checkout(ds)
            self.test_muller_delete_branch(ds)
            self.test_muller_diff(ds)
            self.test_muller_ffmerge(ds)
            self.test_detect_conflict(ds)
            self.test_3_way_merge(ds)
            self.log_and_print("[MULLER] All tests completed successfully")
        except Exception as e:
            self.log_and_print(f"[MULLER] Error during testing: {str(e)}")
            import traceback
            traceback.print_exc()
            if self.log_file:
                traceback.print_exc(file=self.log_file)

    def print_summary(self):
        separator = "=" * 100
        self.log_and_print("\n" + separator)
        self.log_and_print("MULLER VERSION CONTROL BENCHMARK RESULTS")
        self.log_and_print(separator)

        features = [
            ('view_versions', 'View All Versions'),
            ('create_branch', 'Create Branch'),
            ('checkout_branch', 'Checkout Branch'),
            ('delete_branch', 'Delete Branch'),
            ('diff', 'Diff Between Versions'),
            ('ff_merge', 'Fast-Forward Merge'),
            ('detect_conflict', 'Detect Merge Conflict'),
            ('3_way_merge', '3-Way Merge')
        ]

        header = f"\n{'Feature':<30} {'Time (ms)':<15} {'Files':<10}"
        self.log_and_print(header)
        self.log_and_print("-" * 60)

        for key, name in features:
            time_val = self.results['MULLER'].get(key, 'N/A')
            time_str = f"{time_val:.3f}" if isinstance(time_val, float) else time_val

            io_stat = self.io_stats['MULLER'].get(key, {})
            files = io_stat.get('files_count', 'N/A')

            line = f"{name:<30} {time_str:<15} {files:<10}"
            self.log_and_print(line)

        self.log_and_print(separator)
        self.log_and_print(f"\nLog file saved to: {self.log_path}")

    def run_all_tests(self):
        self.open_log()
        try:
            self.log_and_print("Starting MULLER Version Control Performance Benchmark")
            self.log_and_print("=" * 100)

            self.run_muller_tests()
            self.print_summary()

            self.log_and_print("\nBenchmark completed!")
        finally:
            self.close_log()


if __name__ == "__main__":
    import sys

    log_dir = sys.argv[1] if len(sys.argv) > 1 else "./logs"

    benchmark = VersionControlBenchmark(log_dir=log_dir)
    benchmark.run_all_tests()