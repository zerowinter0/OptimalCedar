"""
Main module that provides a CLI interface to run eval for cedar datasets.
"""

import argparse
import json
import importlib
import logging
import os
import sys
import torch

from pathlib import Path

from evaluation.cedar_utils import CedarEvalSpec

from evaluation.profiler import Profiler


logger = logging.getLogger(__name__)


def import_module_from_path(module_path: str):
    module_name = os.path.basename(module_path).rstrip(".py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _get_profiler(dataset_file: str, dataset_func: str, spec: CedarEvalSpec):
    target_file = str(Path(dataset_file).resolve())
    module = import_module_from_path(target_file)

    dataset_getter = getattr(module, dataset_func)
    dataset = dataset_getter(spec)

    # except ModuleNotFoundError as e:
    #     print(f"Could not import module {dataset_file}: {str(e)}")
    #     return
    # except AttributeError as e:
    #     print(f"Could not find function {dataset_func}: {str(e)}")
    #     return

    return Profiler(
        dataset,
        spec.num_epochs,
        spec.num_total_samples,
        spec.batch_size,
        spec.iteration_time,
    )


def write_output(
    dataset_file: str,
    dataset_func: str,
    spec: CedarEvalSpec,
    output_loc: str,
):
    profiler = _get_profiler(dataset_file, dataset_func, spec)
    if not profiler:
        raise RuntimeError("Could not create profiler.")

    if output_loc == "":
        raise RuntimeError("Invalid output location")

    profiler.write_output(output_loc)


def run_profiler(
    dataset_file: str,
    dataset_func: str,
    spec: CedarEvalSpec,
    results_path: str,
):
    """
    Run profiler on provided dataset.

    Args:
        target_file: Path to file defining dataset
        dataset_func: Name of function that returns an iterable or
                            iterdatapipe over a dataset
        spec: Specification of profiler details
        save_loc: Path to json file where profiler output
                        should be saved. Data will be appended
                        to file if save_loc is not the empty string.
                        If dataset_kwargs contains any value for key
                        'overwrite' then the save_loc file gets overwritten.
    """
    profiler = _get_profiler(dataset_file, dataset_func, spec)
    if not profiler:
        raise RuntimeError("Could not create profiler.")

    profiler.run()

    # Save result to specified location
    # NOTE: Printed results are scaled by 10^6 compared to saved results.
    if results_path != "":
        try:
            results = profiler.get_results()

            with open(results_path, "w") as f:
                f.write(json.dumps(results))
                f.write("\n")
            logger.info(f"Appended profiler results to: {results_path}")
        except Exception as e:
            raise Exception(f"Could not save profiler run in file: {str(e)}")


def create_spec(args: argparse.Namespace) -> CedarEvalSpec:
    extra_kwargs = {}
    if args.dataset_kwargs:
        pairs = args.dataset_kwargs.split(",")
        for pair in pairs:
            res = pair.split("=")
            if len(res) == 1:
                key = res[0]
                value = None
            elif len(res) == 2:
                key = res[0]
                value = res[1]
            else:
                raise ValueError(
                    f"Inproperly formatted arg {args.dataset_kwargs}"
                )
            extra_kwargs[key] = value

    return CedarEvalSpec(
        batch_size=args.batch_size,
        num_total_samples=args.num_total_samples,
        num_epochs=args.num_epochs,
        config=args.master_feature_config,
        kwargs=extra_kwargs,
        use_ray=args.use_ray,
        ray_ip=args.ray_ip,
        iteration_time=args.iteration_time,
        profiled_stats=args.profiled_stats,
        run_profiling=args.run_profiling,
        disable_optimizer=args.disable_optimizer,
        disable_controller=args.disable_controller,
        disable_prefetch=args.disable_prefetch,
        disable_offload=args.disable_offload,
        disable_parallelism=args.disable_parallelism,
        disable_reorder=args.disable_reorder,
        disable_fusion=args.disable_fusion,
        disable_caching=args.disable_caching,
        use_my_optimizer=args.use_my_optimizer,
        generate_plan=args.generate_plan,
        reorder_timeout_sec=args.reorder_timeout_sec,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluation Runner")
    parser.add_argument(
        "--dataset_file",
        type=str,
        help="Path to Python file defining dataset.",
        required=True,
    )
    parser.add_argument(
        "--dataset_func",
        type=str,
        default="get_dataset",
        help="Name of function that returns a dataset.",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=1,
        help="Batch size. Benchmarks may not always implement batching.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of epochs to process.",
    )
    parser.add_argument(
        "--num_total_samples",
        type=int,
        help="Number of samples to process in total. If not set, process all.",
    )
    parser.add_argument(
        "--log_level",
        "-l",
        type=str,
        default="INFO",
        help="Minimum level of logs to display. Levels: NOTSET, DEBUG, INFO,\
            WARNING, ERROR, CRITICAL",
    )
    parser.add_argument(
        "--results_path",
        "-s",
        type=str,
        default="",
        help="Indicate path to JSON file to save the eval results.",
    )
    parser.add_argument(
        "--dataset_kwargs",
        type=str,
        help='Extra kwargs to pass to dataset func.\
              Pass as one string, e.g. "split=train,path=train/dir1"',
    )
    parser.add_argument(
        "--write_output",
        type=str,
        default="",
        help="If set, save the output of the data pipeline. \
            Does not run the profiler.",
    )
    parser.add_argument(
        "--master_feature_config",
        type=str,
        help="If path specified, use this feature config for all features.",
    )
    parser.add_argument(
        "--use_ray",
        action="store_true",
        help="Enable the use of Ray for processing.",
    )
    parser.add_argument(
        "--ray_ip",
        type=str,
        default="",
        help="If use_ray is set, the Ray Cluster IP to connect to. \
            If not provided, default to launching a local ray instance",
    )
    parser.add_argument(
        "--iteration_time",
        type=float,
        help="If set, sleep the profiler for <iteration_time> seconds \
            for each iteration.",
    )
    parser.add_argument(
        "--profiled_stats",
        type=str,
        default="",
        help="Path to profiled data (as a YAML file) to use for optimization.",
    )
    parser.add_argument(
        "--run_profiling",
        action="store_true",
        help="If set, simply run profiling and return. "
        "Stores profiled results at the path provided by profiled_stats.",
    )
    parser.add_argument(
        "--allow_torch_parallelism",
        action="store_true",
        help="Set this to allow torch to use multiple threads for "
        "inter/intra op parallelism. With multiprocessing, enabling this "
        "usuall reduces throughput due to thread contention.",
    )
    parser.add_argument(
        "--disable_optimizer",
        action="store_true",
        help="Disable the optimizer for this run",
    )
    parser.add_argument(
        "--disable_controller",
        action="store_true",
        help="Disable the controller for this run",
    )
    parser.add_argument(
        "--disable_prefetch",
        action="store_true",
        help="If optimizer is enabled, disable the prefetch optimization",
    )
    parser.add_argument(
        "--disable_offload",
        action="store_true",
        help="If optimizer is enabled, disable the offload optimization",
    )
    parser.add_argument(
        "--disable_parallelism",
        action="store_true",
        help="If optimizer is enabled, disable the ability to use multiple"
        " local workers",
    )
    parser.add_argument(
        "--disable_reorder",
        action="store_true",
        help="If optimizer is enabled, disable the ability to reorder pipes",
    )
    parser.add_argument(
        "--disable_fusion",
        action="store_true",
        help="If optimizer is enabled, disable the ability to fuse pipes",
    )
    parser.add_argument(
        "--disable_caching",
        action="store_true",
        help="If optimizer is enabled, disable the ability to reorder pipes",
    )
    parser.add_argument(
        "--use_my_optimizer",
        type=int,
        choices=[0, 1, 2, 3, 4, 5, 6],
        default=0,
        help="Optimizer selector: 0 original Cedar, 6 Cedar joint enumeration.",
    )
    parser.add_argument(
        "--reorder_timeout_sec",
        type=float,
        default=None,
        help="Maximum wall-clock seconds for reorder enumeration.",
    )
    parser.add_argument(
        "--generate_plan",
        action="store_true",
        help="If optimizer is enabled, generate the optimized plan and exit",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Indicate the number of expected samples. Should be used with caching.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper())

    spec = create_spec(args)

    # Set torch parallelism
    if not args.allow_torch_parallelism:
        logger.warning("Setting torch threads to 1")
        torch.set_num_threads(1)
        # torch.set_num_interop_threads(1)

    if args.write_output:
        write_output(
            args.dataset_file, args.dataset_func, spec, args.write_output
        )
    else:
        run_profiler(
            args.dataset_file, args.dataset_func, spec, args.results_path
        )


if __name__ == "__main__":
    main()
