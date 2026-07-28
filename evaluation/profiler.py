import os
import time
import logging
from typing import Optional, Iterable

logger = logging.getLogger(__name__)

PROGRESS_UPDATE_TIME_SEC = 5


class Timer:
    def __init__(self):
        self._start = None
        self._end = None

    def __enter__(self):
        self._start = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end = time.perf_counter()

    def reset(self):
        self._start = time.perf_counter()

    def delta(self):
        if self._start is None or self._end is None:
            raise RuntimeError()
        return self._end - self._start


class Profiler:
    """
    Profiler used for measuring data loading performance.

    Accepts any iterable.
    """

    def __init__(
        self,
        dataset: Iterable,
        num_epochs: int = 1,
        num_total_samples: Optional[int] = None,
        batch_size: int = 1,
        iteration_time: Optional[float] = None,
    ) -> None:
        self.load_time = 0
        self.epoch_run_times = []
        self.epoch_num_samples = []
        self.num_epochs = num_epochs
        self.num_total_samples = num_total_samples
        self.batch_size = batch_size
        self.iteration_time = iteration_time
        # clear the page cache to get cleanest results
        os.system('sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"')

        self.dataset = dataset

    def write_output(self, file: str):
        """
        Saves up to num_total_samples (or the entire epoch) to a file.
        """
        with open(file, "a") as f:
            curr_total_samples = 0
            for _ in range(self.num_epochs):
                curr_epoch_samples = 0
                for x in self.dataset:
                    curr_total_samples += self.batch_size
                    curr_epoch_samples += self.batch_size
                    if (
                        self.num_total_samples
                        and curr_total_samples >= self.num_total_samples
                    ):
                        break
                    f.write(str(x))
                    f.write("\n")
                    try:
                        f.write(f"Tensor size: {str(x.size())}\n")
                    except Exception:
                        try:
                            f.write(f"Tensor size: {str(x.shape)}\n")
                        except Exception:
                            try:
                                f.write(
                                    f"Tensor size: {str(x['image'].size())}\n"
                                )
                            except Exception:
                                pass
                        pass

    def run(self) -> None:
        """
        Run the profiler.

        NOTE: The reported number of samples may be inaccurate for
            tail batches (e.g., incomplete last batch in the epoch)
        """
        curr_total_samples = 0
        print_time = time.time() + PROGRESS_UPDATE_TIME_SEC

        for epoch in range(self.num_epochs):
            if (
                self.num_total_samples
                and curr_total_samples >= self.num_total_samples
            ):
                break
            curr_epoch_samples = 0
            timer = Timer()
            with timer:
                if self.iteration_time is not None:
                    curr_it_time = time.time()
                for i, x in enumerate(self.dataset):
                    # We want to measure steady state latency, so reset
                    # timer upon getting first batch to ignore
                    # startup costs
                    if i == 1:
                        timer.reset()

                    curr_total_samples += self.batch_size
                    curr_epoch_samples += self.batch_size

                    if (
                        self.num_total_samples
                        and curr_total_samples >= self.num_total_samples
                    ):
                        break

                    curr_time = time.time()
                    if curr_time >= print_time:
                        self._print_status(
                            epoch,
                            curr_total_samples,
                            curr_epoch_samples,
                        )
                        print_time = curr_time + PROGRESS_UPDATE_TIME_SEC

                    if self.iteration_time is not None:
                        delta = curr_time - curr_it_time
                        sleep_time = self.iteration_time - delta
                        # curr_it_time = curr_time
                        if sleep_time > 0:
                            time.sleep(sleep_time)

                        if i % 2 == 0:
                            curr_it_time = time.time()

            epoch_time = timer.delta()
            self._print_status(
                epoch,
                curr_total_samples,
                curr_epoch_samples,
                epoch_time,
            )

            self.epoch_run_times.append(timer.delta())
            self.epoch_num_samples.append(curr_epoch_samples)

        self.print_results()

    def print_results(self):
        total_run_time = sum(self.epoch_run_times)
        total_samples = sum(self.epoch_num_samples)

        print("======Epoch time breakdowns======")
        for idx, epoch in enumerate(self.epoch_run_times):
            print(
                "Epoch [{}/{}]: {:4.3f}s".format(
                    idx, len(self.epoch_run_times) - 1, epoch
                )
            )

        print("======Benchmark Summary======")
        print(
            (
                "Total time: {:4.3f}s\n"
                + "\tnum samples: {}\n"
                + "\ttime per sample: {:4.3f}us"
            ).format(
                total_run_time,
                total_samples,
                (total_run_time / total_samples) * 1e6,
            )
        )

    def get_results(self):
        """
        Returns a dict with profiled results.
        """
        return {
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "num_total_samples": self.num_total_samples,
            "epoch_run_times": self.epoch_run_times,
            "epoch_num_samples": self.epoch_num_samples,
        }

    def close(self) -> None:
        """Release resources held by the underlying dataset."""
        close = getattr(self.dataset, "close", None)
        if callable(close):
            close()
            return
        exit_dataset = getattr(self.dataset, "_exit", None)
        if callable(exit_dataset):
            exit_dataset()

    def _print_status(
        self,
        epoch: int,
        curr_total_samples: int,
        curr_epoch_samples: int,
        epoch_time: Optional[int] = None,
    ) -> None:
        print(
            "Epoch [{}/{}]: [{}/{}] epoch/total samples".format(
                epoch,
                self.num_epochs - 1,
                curr_epoch_samples,
                curr_total_samples,
            )
        )
        if epoch_time:
            print("     Epoch time: {:4.3f}s".format(epoch_time))
