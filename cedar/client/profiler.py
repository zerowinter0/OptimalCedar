import logging
import math
import statistics
import threading
import time
from typing import Dict, Deque, Optional, Tuple
from collections import deque

from cedar.pipes import DataSample
from cedar.compose import Feature
from .logger import DataSetLogger
from .constants import (
    MAX_HISTORY,
    THROUGHPUT_LOG_TIME_SEC,
)

logger = logging.getLogger(__name__)


class FeatureProfiler:
    """
    Profiler meant to profile a given feature.

    Args:
        feature: Feature to profile
        logger: DataSetLogger to write logs
        profile_mode: True to enable profile mode.
    """

    def __init__(
        self,
        feature: Feature,
        logger: Optional[DataSetLogger] = None,
        profile_mode: bool = False,
    ):
        self.feature = feature
        # NOTE: Assumes batch size is static
        self.batch_size = self.feature.get_batch_size()

        # List of calculated latencies to yield a sample for each pipe, in ns
        self.latencies: Dict[int, Deque[int]] = {}
        # Wall-clock counterpart collected for offline profile artifacts.
        self.wall_latencies: Dict[int, Deque[int]] = {}
        # List of buffer lengths for each async pipe
        self.buffer_sizes: Dict[int, Deque[int]] = {}
        # List of data sizes for each async pipe
        self.data_sizes: Dict[int, Deque[Tuple[float, float]]] = {}
        # Paired legal-input observations used by the optional operator
        # scaling classifier: (input bytes per record, wall latency per
        # record). Keeping the pair avoids aligning independent aggregates.
        self.compute_scaling_observations: Dict[
            int, Deque[Tuple[float, float]]
        ] = {}
        # sorted pipes by ID, in topological order
        self.ds_logger = logger

        self._lock = threading.Lock()

        # Lock for sample counter
        self._sample_count = 0
        self._prev_throughput_log_time = time.time()
        self._prev_throughput_sample_count = 0

        self.profile_mode = profile_mode

    def update_ds(self, ds: DataSample):
        """
        Update the profiler with a data sample
        """
        # Track throughput
        self._sample_count += 1
        if not ds.do_trace:
            return
        # Log throughput if time
        if (
            self.ds_logger
            and time.time()
            >= self._prev_throughput_log_time + THROUGHPUT_LOG_TIME_SEC
        ):
            curr_time = time.time()
            # delta_sample_count = (
            #     self._sample_count - self._prev_throughput_sample_count
            # )
            # tput = (self.batch_size * delta_sample_count) / (
            #     curr_time - self._prev_throughput_log_time
            # )
            self._prev_throughput_log_time = curr_time
            self._prev_throughput_sample_count = self._sample_count

            # self.ds_logger.log(
            #     "Throughput [{}s window]: {}".format(
            #         THROUGHPUT_LOG_TIME_SEC, tput
            #     )
            # )

        if self.feature.has_changed_dataflow():
            with self._lock:
                if self.profile_mode:
                    maxlen = None
                else:
                    maxlen = MAX_HISTORY
                # Clear latencies
                self.latencies = {
                    p_id: deque(maxlen=maxlen)
                    for p_id in self.feature.physical_pipes.keys()
                }
                if self.profile_mode:
                    self.wall_latencies = {
                        p_id: deque(maxlen=maxlen)
                        for p_id in self.feature.physical_pipes.keys()
                    }
                self.buffer_sizes = {
                    p_id: deque(maxlen=maxlen)
                    for p_id in self.feature.physical_pipes.keys()
                }
                if self.profile_mode:
                    self.data_sizes = {
                        p_id: deque(maxlen=maxlen)
                        for p_id in self.feature.physical_pipes.keys()
                    }
                    self.compute_scaling_observations = {
                        p_id: deque(maxlen=maxlen)
                        for p_id in self.feature.physical_pipes.keys()
                    }
        with self._lock:
            self._update_latencies(ds)
            if self.profile_mode:
                self._update_wall_latencies(ds)
            self._update_buffer_sizes(ds)
            if self.profile_mode:
                self._update_data_sizes(ds)
                self._update_compute_scaling_observations(ds)

    def _update_latencies(self, ds: DataSample) -> None:
        """
        Given ds, update list of latencies

        TODO: a lot of error handling
        """
        if len(ds.trace_order) < 2:
            return

        # First pipe (source) assumed to have no latency
        # NOTE: Only works for linear pipes now...
        log_dict = {}
        for idx, p_id in enumerate(ds.trace_order[1:], start=1):
            # TODO: handle error if p_id not found
            latency = (
                ds.trace_dict[p_id] - ds.trace_dict[ds.trace_order[idx - 1]]
            )

            # Normalize to per-sample
            latency = latency / ds.size_dict[p_id]

            self.latencies[p_id].append(latency)
            log_dict[p_id] = latency / 1e3

        # if self.ds_logger:
        #     self.ds_logger.log("====Trace====")
        #     self.ds_logger.log(
        #         "\tPIPE LATENCY per sample (us): " + str(log_dict)
        #     )
        #     self.ds_logger.log("\tPIPE SIZES: " + str(ds.size_dict))
        #     self.ds_logger.log("\tPIPE ORDER: " + str(ds.trace_order))

    def _update_buffer_sizes(self, ds: DataSample) -> None:
        log_dict = {}
        for p_id, buf_size in ds.buffer_size_dict.items():
            self.buffer_sizes[p_id].append(buf_size)
            log_dict[p_id] = buf_size
        # if self.ds_logger:
        #     self.ds_logger.log("\tBUF SIZE: " + str(log_dict))

    def _update_wall_latencies(self, ds: DataSample) -> None:
        if (
            len(ds.trace_order) < 2
            or ds.wall_trace_dict is None
            or ds.wall_trace_resume_dict is None
        ):
            return
        for idx, p_id in enumerate(ds.trace_order[1:], start=1):
            prev_p_id = ds.trace_order[idx - 1]
            if (
                p_id not in ds.wall_trace_dict
                or prev_p_id not in ds.wall_trace_resume_dict
            ):
                continue
            latency = (
                ds.wall_trace_dict[p_id]
                - ds.wall_trace_resume_dict[prev_p_id]
            ) / ds.size_dict[p_id]
            self.wall_latencies[p_id].append(latency)

    def _update_data_sizes(self, ds: DataSample) -> None:
        if len(ds.trace_order) < 2:
            return

        if ds.data_size_dict is None:
            raise RuntimeError("Datasample contains no data size dict")

        for idx, p_id in enumerate(ds.trace_order[1:], start=1):
            prev_p_id = ds.trace_order[idx - 1]

            # Normalize to per sample. Preserve Cedar's calculation when
            # tracing is complete; only fall back when size tracing failed.
            prev_sample_size = ds.size_dict.get(prev_p_id, 1) or 1
            input_size = ds.data_size_dict.get(prev_p_id, 0) / prev_sample_size
            curr_data_size = ds.data_size_dict.get(p_id)
            if curr_data_size is None:
                curr_data_size = input_size * (ds.size_dict.get(p_id, 1) or 1)
                logger.warning(
                    "Missing traced data size for pipe %s; using input "
                    "size %.6f as output-size fallback.",
                    p_id,
                    input_size,
                )
            output_size = curr_data_size / (ds.size_dict.get(p_id, 1) or 1)
            self.data_sizes[p_id].append((input_size, output_size))

    def _update_compute_scaling_observations(self, ds: DataSample) -> None:
        """Collect paired timings over naturally different legal inputs."""
        if (
            len(ds.trace_order) < 2
            or ds.data_size_dict is None
            or ds.wall_trace_dict is None
            or ds.wall_trace_resume_dict is None
        ):
            return
        for idx, p_id in enumerate(ds.trace_order[1:], start=1):
            prev_p_id = ds.trace_order[idx - 1]
            if (
                prev_p_id not in ds.data_size_dict
                or p_id not in ds.wall_trace_dict
                or prev_p_id not in ds.wall_trace_resume_dict
            ):
                continue
            input_records = ds.size_dict.get(prev_p_id, 1) or 1
            output_records = ds.size_dict.get(p_id, 1) or 1
            input_bytes = ds.data_size_dict[prev_p_id] / input_records
            latency_ns = (
                ds.wall_trace_dict[p_id]
                - ds.wall_trace_resume_dict[prev_p_id]
            ) / output_records
            if input_bytes > 0 and latency_ns > 0:
                self.compute_scaling_observations[p_id].append(
                    (float(input_bytes), float(latency_ns))
                )

    def infer_compute_scaling(self) -> Dict[int, Dict[str, float]]:
        """Classify operators from low/high input-size timing strata.

        The classifier deliberately abstains when profiling did not observe
        enough legal records or enough input-size variation. An abstention is
        represented by ``per_data`` with zero confidence, matching Cedar's
        documented default rather than inventing a per-record label.
        """
        result = {}
        with self._lock:
            observations = {
                p_id: list(values)
                for p_id, values in self.compute_scaling_observations.items()
            }
        for p_id, values in observations.items():
            values = [
                pair for pair in values
                if all(math.isfinite(value) and value > 0 for value in pair)
            ]
            values.sort(key=lambda pair: pair[0])
            n = len(values)
            if n < 12:
                result[p_id] = {
                    "scaling": "per_data",
                    "confidence": 0.0,
                    "n_observations": n,
                    "reason": "insufficient_observations",
                }
                continue

            group_size = max(4, n // 3)
            low = values[:group_size]
            high = values[-group_size:]
            low_size = statistics.median(pair[0] for pair in low)
            high_size = statistics.median(pair[0] for pair in high)
            low_latency = statistics.median(pair[1] for pair in low)
            high_latency = statistics.median(pair[1] for pair in high)
            size_ratio = high_size / low_size
            latency_ratio = high_latency / low_latency
            if size_ratio < 1.5:
                result[p_id] = {
                    "scaling": "per_data",
                    "confidence": 0.0,
                    "n_observations": n,
                    "size_ratio": size_ratio,
                    "latency_ratio": latency_ratio,
                    "reason": "insufficient_size_variation",
                }
                continue

            elasticity = math.log(max(latency_ratio, 1e-12)) / math.log(
                size_ratio
            )
            per_data = latency_ratio >= 1.25 and elasticity >= 0.35
            # Confidence is distance from the decision boundary, capped to a
            # portable [0, 1] diagnostic rather than a probabilistic claim.
            if per_data:
                confidence = min(1.0, max(0.0, (elasticity - 0.35) / 0.65))
            else:
                confidence = min(1.0, max(0.0, (0.35 - elasticity) / 0.35))
            predicted_scaling = "per_data" if per_data else "per_record"
            confident = confidence >= 0.5
            result[p_id] = {
                # Low-confidence observations abstain to the documented
                # per-data default. Keep the raw prediction for evaluation.
                "scaling": predicted_scaling if confident else "per_data",
                "predicted_scaling": predicted_scaling,
                "confidence": confidence,
                "n_observations": n,
                "size_ratio": size_ratio,
                "latency_ratio": latency_ratio,
                "elasticity": elasticity,
                "reason": "classified" if confident else "low_confidence",
            }
        return result

    def calculate_avg_latency_per_sample(self) -> Dict[int, float]:
        """
        Returns a dict mapping each pipe ID to the average traced latency
        required to process a sample.
        """
        with self._lock:
            res = {
                k: sum(v) / len(v)
                for k, v in self.latencies.items()
                if len(v) > 0
            }
        return res

    def calculate_avg_wall_latency_per_sample(self) -> Dict[int, float]:
        """Return per-pipe wall-clock latency in nanoseconds per sample."""
        with self._lock:
            return {
                k: sum(v) / len(v)
                for k, v in self.wall_latencies.items()
                if len(v) > 0
            }

    def calculate_avg_buffer_size(self) -> Dict[int, float]:
        """
        Returns a dict mapping each pipe ID to the average traced
        buffer size for asynchronous pipes.
        """
        with self._lock:
            res = {
                k: sum(v) / len(v)
                for k, v in self.buffer_sizes.items()
                if len(v) > 0
            }
        return res

    def get_sample_count(self) -> int:
        return self._sample_count

    def get_batch_size(self) -> int:
        return self.batch_size

    def calculate_avg_data_size(
        self,
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Returns a tuple, with each element mapping each pipe ID to the average
        trace (input size, output size) of data for the pipe
        """
        with self._lock:
            input_sizes = {
                k: sum([w[0] for w in v]) / len(v)
                for k, v in self.data_sizes.items()
                if len(v) > 0
            }
            output_sizes = {
                k: sum([w[1] for w in v]) / len(v)
                for k, v in self.data_sizes.items()
                if len(v) > 0
            }
        return (input_sizes, output_sizes)
