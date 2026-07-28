"""
Main module that provides a CLI interface to run eval for torch datasets.
"""

import argparse
import json
import importlib
import logging
import os
import sys

from pathlib import Path

from evaluation.torch_utils import TorchEvalSpec

from evaluation.profiler import Profiler


logger = logging.getLogger(__name__)


def import_module_from_path(module_path: str):
    module_path_obj = Path(module_path).resolve()
    try:
        rel_path = module_path_obj.relative_to(Path.cwd().resolve())
        module_name = ".".join(rel_path.with_suffix("").parts)
    except ValueError:
        module_name = os.path.basename(module_path).removesuffix(".py")
    spec = importlib.util.spec_from_file_location(module_name, str(module_path_obj))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _get_profiler(dataset_file: str, dataset_func: str, spec: TorchEvalSpec):
    try:
        target_file = str(Path(dataset_file).resolve())
        module = import_module_from_path(target_file)

        dataset_getter = getattr(module, dataset_func)
        dataset = dataset_getter(spec)

    except ModuleNotFoundError as e:
        print(f"Could not import module {dataset_file}: {str(e)}")
        return
    except AttributeError as e:
        print(f"Could not find function {dataset_func}: {str(e)}")
        return

    return Profiler(
        dataset, spec.num_epochs, spec.num_total_samples, spec.batch_size
    )


def write_output(
    dataset_file: str,
    dataset_func: str,
    spec: TorchEvalSpec,
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
    spec: TorchEvalSpec,
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


def create_spec(args: argparse.Namespace) -> TorchEvalSpec:
    return TorchEvalSpec(
        args.batch_size,
        args.num_workers,
        args.num_epochs,
        args.num_total_samples,
        args.iteration_time,
        args.dataset_kwargs,
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
        "--write_output",
        type=str,
        default="",
        help="If set, save the output of the data pipeline. \
            Does not run the profiler.",
    )
    parser.add_argument(
        "--iteration_time",
        type=float,
        help="If set, sleep the profiler for <iteration_time> seconds \
            for each iteration.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of torch dataloader workers to use. If not set, use"
        " the main process.",
        default=0,
    )
    parser.add_argument(
        "--dataset_kwargs",
        type=json.loads,
        default={},
        help="JSON object passed to the workload through TorchEvalSpec.kwargs.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper())

    spec = create_spec(args)

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
