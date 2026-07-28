# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved
import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from enum import Enum

import tensorflow as tf
import yaml

from tensorflow import keras
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import fastflow as ff
from fastflow import utils as ff_utils


class OffloadingType(Enum):
    FASTFLOW = "ff"  # fastflow
    TF_DSR = "tf-dsr"  # tf-dsr
    TF_DSR_ALL = "tf-dsr-all"  # tf-dsr
    TF_DSLR = "tf-dslr"  # tf-dslr
    TF_DSLR_ALL = "tf-dslr-all"  # tf-dslr
    DALI = "dali"  # dali
    TENSORFLOW = "tf"  # tensorflow


class GPUType(Enum):
    SINGLE = "single"  # single-gpu training
    MULTI = "multi"  # multi-gpu training


class App:
    """
    Interface for Applications.
    Applications must inherit this class.
    Ex) class CtcAsr(App):
    """

    def __init__(self, args, config):
        """
        :param args:  command line arguments
        :param config:  yaml config
        """
        self.args = args
        self.config = config

    def create_model(self):
        """
        Create a model.
        """
        raise NotImplementedError

    def create_dataset(self, num_parallel):
        """
        Create train dataset input pipeline
        :param num_parallel: number of parallelism
        """
        raise NotImplementedError

    def create_valid_dataset(self, num_parallel):
        """
        Create valid dataset input pipeline
        :param num_parallel:  number of parallelism
        """
        raise NotImplementedError

    def create_manual_offloaded_dataset(self, num_parallel):
        """
        Create the train dataset input pipeline with manual offloading.
        :param num_parallel: number of parallelism
        """
        raise NotImplementedError

    def create_manual_offloaded_valid_dataset(self, num_parallel):
        """
        Create the validation dataset input pipeline with manual offloading.
        :param num_parallel: number of parallelism
        """
        raise NotImplementedError

    def create_all_offload_dataset(self, num_parallel):
        """
        Create the train dataset input pipeline with manual offloading all operations.
        :param num_parallel: number of parallelism
        """
        raise NotImplementedError

    def create_all_offload_valid_dataset(self, num_parallel):
        """
        Create the valid dataset input pipeline with manual offloading all operations.
        :param num_parallel: number of parallelism
        """
        raise NotImplementedError

    def create_dali_dataset(self, num_parallel):
        """
        Create dali train dataset input pipeline
        :param num_parallel: number of parallelism
        """
        raise NotImplementedError

    def create_dali_valid_dataset(self, num_parallel):
        """
        Create dali valid dataset input pipeline
        :param num_parallel:  number of parallelism
        """
        raise NotImplementedError

    def callbacks(self):
        """
        Return callbacks for model.fit()
        :return:
        """
        return []

    def steps_per_epoch_for_dali(self):
        """
        Return steps_per_epoch used in model.fit() for DALI
        :return: steps_per_epoch
        """
        raise NotImplementedError


class AppRunner:
    def __init__(self, app, args, config):
        self.app = app
        self.args = args
        self.config = config

    def run(self):
        result_callback = ResultCallback(
            self.args.results_path,
            self.args.offloading_type.value,
        )

        if self.args.parallel > 0:
            parallel = self.args.parallel
        else:
            parallel = tf.data.AUTOTUNE

        # Dataset setting
        if self.args.offloading_type in (OffloadingType.TF_DSR,
                                         OffloadingType.TF_DSLR):
            self.config.partial_offload_enabled = False
            dataset = self.app.create_manual_offloaded_dataset(parallel)
            valid_dataset = self.app.create_manual_offloaded_valid_dataset(
                parallel
            )
        elif self.args.offloading_type in (OffloadingType.TF_DSR_ALL,
                                           OffloadingType.TF_DSLR_ALL):
            self.config.partial_offload_enabled = False
            dataset = self.app.create_all_offload_dataset(parallel)
            valid_dataset = self.app.create_all_offload_valid_dataset(
                parallel
            )
        elif self.args.offloading_type is OffloadingType.FASTFLOW:
            self.config.partial_offload_enabled = True
            dataset = self.app.create_dataset(parallel)
            valid_dataset = self.app.create_valid_dataset(parallel)
        elif self.args.offloading_type is OffloadingType.TENSORFLOW:
            dataset = self.app.create_dataset(parallel)
            valid_dataset = self.app.create_valid_dataset(parallel)
        elif self.args.offloading_type is OffloadingType.DALI:
            dataset = self.app.create_dali_dataset(parallel)
            valid_dataset = self.app.create_dali_valid_dataset(parallel)
        else:
            raise RuntimeError("Invalid offloading type: "
                               + self.args.offloading_type)

        # Model setting
        if self.args.gpu_type is GPUType.SINGLE:
            model = self.app.create_model()
        else:
            # Create a MirroredStrategy for multi-gpu training.
            strategy = tf.distribute.MultiWorkerMirroredStrategy()
            print("Number of devices: {}".format(
                strategy.num_replicas_in_sync
            ))
            with strategy.scope():
                model = self.app.create_model()

        # Model running
        if self.args.offloading_type is OffloadingType.FASTFLOW:
            # Run fastflow model for auto offloading
            steps_per_epoch = None
            if self.args.num_samples is not None:
                steps_per_epoch = (
                    self.args.num_samples + self.args.batch - 1
                ) // self.args.batch
            model.fit(x=dataset,
                      validation_data=valid_dataset,
                      epochs=self.args.epochs,
                      callbacks=self.app.callbacks() + [result_callback],
                      steps_per_epoch=steps_per_epoch,
                      auto_offload_conf=self.config)
        elif self.args.offloading_type is OffloadingType.DALI:
            # Run model with GPU offloading with DALI
            steps_per_epoch = self.app.steps_per_epoch_for_dali()
            keras.models.Model.fit(model,
                                   x=dataset,
                                   # validation_data=valid_dataset,
                                   epochs=self.args.epochs,
                                   callbacks=self.app.callbacks()
                                             + [ff_utils.EpochTimeCallback(),
                                                result_callback],
                                   steps_per_epoch=steps_per_epoch)
        else:
            # Run the original model not to perform auto-offloading.
            keras.models.Model.fit(model,
                                   x=dataset,
                                   validation_data=valid_dataset,
                                   epochs=self.args.epochs,
                                   callbacks=self.app.callbacks()
                                             + [ff_utils.EpochTimeCallback(),
                                                result_callback])


class ResultCallback(tf.keras.callbacks.Callback):
    """Write machine-readable epoch timing without changing model behavior."""

    def __init__(self, results_path, offloading_type):
        super().__init__()
        self.results_path = results_path
        self.offloading_type = offloading_type
        self.epoch_times = []
        self._epoch_start = None

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.perf_counter() - self._epoch_start)

    def on_train_end(self, logs=None):
        if not self.results_path:
            return
        path = Path(self.results_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "system": "fastflow",
            "offloading_type": self.offloading_type,
            "epoch_times_sec": self.epoch_times,
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def get_arguments():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument("app_file_path", type=str)
    parser.add_argument("data_prefix", type=str)
    parser.add_argument("offloading_type", type=OffloadingType,
                        choices=[OffloadingType.TENSORFLOW,
                                 OffloadingType.TF_DSR,
                                 OffloadingType.TF_DSR_ALL,
                                 OffloadingType.TF_DSLR,
                                 OffloadingType.TF_DSLR_ALL,
                                 OffloadingType.FASTFLOW,
                                 OffloadingType.DALI])
    parser.add_argument("yaml_path", type=str)
    parser.add_argument("--gpu_type", type=GPUType, default=GPUType.SINGLE,
                        choices=[GPUType.SINGLE, GPUType.MULTI])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--parallel", type=int, default=-1)
    parser.add_argument("--num_local_workers", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--results_path", type=str, default="")
    args = parser.parse_args()
    yaml_dict = yaml.load(open(args.yaml_path), Loader=yaml.FullLoader)
    return args, yaml_dict


def get_address():
    # The comparison container uses host networking and all tf.data service
    # workers are local. hostname -I may select a routable host interface whose
    # gRPC traffic is filtered; loopback is deterministic for this topology.
    return os.environ.get("FASTFLOW_LOCAL_ADDRESS", "127.0.0.1")


def launch_local_worker(num_local_workers, dispatcher_addr, worker_base_port):
    print("Launch local worker")
    num_workers = num_local_workers
    workers = []
    for i in range(num_workers):
        port = worker_base_port + i + 1
        w_config = tf.data.experimental.service.WorkerConfig(
            dispatcher_address=dispatcher_addr + ":5000",
            worker_address=get_address() + ":" + str(port),
            port=port)

        worker = tf.data.experimental.service.WorkerServer(w_config)
        workers.append(worker)
    return workers


def launch_remote_worker_processes(
    num_workers, dispatcher_addr, worker_base_port
):
    """Launch workers outside this process so FastFlow classifies them remote.

    FastFlow's custom TensorFlow op identifies local workers through the
    process-local WorkerServer registry, not by comparing IP addresses.  A
    second set of in-process WorkerServer objects is therefore still local and
    makes partial offload fail before producing any samples.
    """
    processes = []
    script_path = str(Path(__file__).resolve())
    for i in range(num_workers):
        port = worker_base_port + i + 1
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    script_path,
                    "--serve-remote-worker",
                    dispatcher_addr,
                    str(port),
                ]
            )
        )
    return processes


def stop_worker_processes(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def wait_for_registered_workers(dispatcher, expected, processes, timeout=30):
    deadline = time.monotonic() + timeout
    while dispatcher._num_workers() < expected:
        failed = [p.returncode for p in processes if p.poll() is not None]
        if failed:
            raise RuntimeError(
                "FastFlow remote worker exited during startup: "
                + repr(failed)
            )
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Timed out waiting for FastFlow workers: "
                f"registered={dispatcher._num_workers()} expected={expected}"
            )
        time.sleep(0.05)


if __name__ == "__main__":
    """
     Arguments
     1) app_file_path: the example app path 
     2) offloading_type: (tf, tf-dsr, tf-dslr, ff)
     3) yaml_path: the yaml config file path fo the example app
     4) gpu_type: ('single', 'multi')
    """
    if len(sys.argv) == 4 and sys.argv[1] == "--serve-remote-worker":
        # Ensure a worker cannot survive an abruptly terminated parent runner.
        try:
            import ctypes

            ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        except (AttributeError, OSError):
            pass
        remote_dispatcher_addr = sys.argv[2]
        remote_port = int(sys.argv[3])
        remote_config = tf.data.experimental.service.WorkerConfig(
            dispatcher_address=remote_dispatcher_addr + ":5000",
            worker_address=get_address() + ":" + str(remote_port),
            port=remote_port,
        )
        remote_worker = tf.data.experimental.service.WorkerServer(
            remote_config
        )
        remote_worker.join()
        sys.exit(0)

    args, yaml_dict = get_arguments()
    yaml_path = args.yaml_path
    app_path = args.app_file_path

    print('Args: ', args)

    # Extract app from an explicit path. APP_CLASS avoids global
    # App.__subclasses__ state leaking across imported workload modules.
    app_file = Path(app_path).resolve()
    spec = importlib.util.spec_from_file_location(
        "fastflow_workload_app",
        app_file,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import FastFlow app: " + str(app_file))
    app_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = app_module
    spec.loader.exec_module(app_module)
    app_class = getattr(app_module, "APP_CLASS", None)
    if app_class is None:
        subclasses = app_module.App.__subclasses__()
        if len(subclasses) != 1:
            raise RuntimeError(
                "FastFlow app must define APP_CLASS or exactly one App "
                "subclass; found " + repr(subclasses)
            )
        app_class = subclasses[0]

    config = ff.FastFlowConfig.from_yaml(yaml_path)

    # Launch local dispatcher and worker for FastFlow
    remote_worker_processes = []
    if args.offloading_type is OffloadingType.FASTFLOW:
        d_config = tf.data.experimental.service.DispatcherConfig(port=5000)
        dispatcher = tf.data.experimental.service.DispatchServer(d_config)
        workers_local_dispatcher = launch_local_worker(args.num_local_workers,
                                                       get_address(), 5000)

    if args.offloading_type in (OffloadingType.TF_DSLR,
                                OffloadingType.TF_DSLR_ALL,
                                OffloadingType.FASTFLOW) \
            and not config.autoscale_enabled:
        remote_worker_processes = launch_remote_worker_processes(
            args.num_local_workers, config.dispatcher_addr, 5500
        )
        if args.offloading_type is OffloadingType.FASTFLOW:
            wait_for_registered_workers(
                dispatcher,
                expected=2 * args.num_local_workers,
                processes=remote_worker_processes,
            )

    if args.offloading_type in (OffloadingType.TF_DSR,
                                OffloadingType.TF_DSLR,
                                OffloadingType.TF_DSR_ALL,
                                OffloadingType.TF_DSLR_ALL,
                                OffloadingType.FASTFLOW) \
            and not config.dispatcher_addr:
        raise RuntimeError("Offloading Type is " + str(args.offloading_type) +
                           " , but dispatcher_addr "
                           "is not configured in the yaml.")

    if args.offloading_type is OffloadingType.DALI:
        config_gpu_mem = ConfigProto()
        config_gpu_mem.gpu_options.per_process_gpu_memory_fraction = 0.8
        session = InteractiveSession(config=config_gpu_mem)

    app = app_class(args, config)

    runner = AppRunner(app, args, config)
    try:
        runner.run()
    finally:
        stop_worker_processes(remote_worker_processes)
