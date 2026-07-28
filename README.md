# cedar

`cedar` is a composable and optimized framework to define
end-to-end data storage and ingestion pipelines for ML workloads.

## Setup
To setup cedar and its dependencies, begin by making sure that you have Python>=3.9 installed, as well as `pip` and `venv`.

We suggest first creating an virtual environment.
Then, simply `pip install` the `requirements.txt` file.

```bash
cd <PATH_TO_REPO>
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Finally, install cedar as a package using pip.
```bash
pip install -e .
```

In the setup instructions below, we refer to the node that you run the main data loading process on as the `local` node, and any distributed workers as `remote` nodes. The instructions assume that you execute commands on the `local` node unless otherwise specified.

### Downloading Datasets
We provide scripts to download datasets for each set of evaluation pipelines.
Simply execute the following scripts.

- `CV`: `python <PATH_TO_REPO>/evaluation/pipelines/simclrv2/download.py`
- `NLP`: `python <PATH_TO_REPO>/evaluation/pipelines/wikitext103/download.py`
- `ASR`: Mozilla requires you to manually register and download the CommonVoice dataset via their website (https://commonvoice.mozilla.org/en/datasets). Specifically, we used the `cv-corpus-15.0-delta-2023-09-08` set. Once you have downloaded the dataset, save all of the raw MP3s locally at `<PATH_TO_REPO>/evaluation/datasets/commonvoice/cv-corpus-15.0-delta-2023-09-08/en/clips`. 

### Distributed Processing
cedar uses Ray Core to distribute preprocessing. To use Ray, you first need
to set up the appropriate permissions and settings on GCP (or whichever cloud provider you use), as detailed
[here](https://docs.ray.io/en/latest/cluster/vms/getting-started.html#vm-cluster-quick-start).

Once Ray is setup, you will have to first create a machine image that Ray can automatically create a cluster with.
Since Ray requires that the same environment is used between nodes in the Ray cluster, the easiest way to do so is to simply clone the VM that you already installed dependencies and downloaded datasets on.

Once you have an image with all dependencies installed, update the respective field in `configs/ray.yaml` (specifically, the `node_config`) to point to your appropriate image.

Additionally, you will need to update the `setup_commands` to point to your respective home directory.

Once the config is setup, you can simply create the Ray Cluster by running the following.

```bash
ray up -y configs/ray.yaml
```

Once you are done with Ray, **REMEMBER TO DESTROY THE CLUSTER**

```bash
ray down -y configs/ray.yaml
```

## Design

### Pipes and Variants
A `Pipe` (see `cedar/cedar/pipes/pipe.py`) represents an operator that is applied to training samples.
With the exception of `Sources` (which are also Pipes, but with no input Pipes), each Pipe takes in one or more input Pipes.
Logically, each Pipe iteratively reads samples from its upstream Pipe(s), performs an operation, and yields the sample to the downstream Pipe(s) (NOTE: multiple inputs/outputs not currently supported).

Each Pipe has multiple `Variants` (`cedar/cedar/pipes/variant.py`), which is a physical implementation of the operation that the Pipe performs.
For example, there may be an `InProcess` Variant which performs processing in the main Python process, an SMP (multiprocessing variant), a distributed Ray Variant, etc.
Each Pipe must implement the InProcess Variant.

Each Variant exposes an iterator interface that yields samples that have operators applied up to (and including) the Pipe itself.
Specifically, each time `next` is called on a Variant, the Variant calls `next` on its upstream Pipe's Variant, processes the sample, and yields it as an output.

To implement a new `Pipe`, you need to do the following:
- First, make a subclass of `Pipe` and call its superclass constructor appropriately.
- Decorate the new Pipe with an `@cedar_pipe` decorator that specifies an `CedarPipeSpec`, which lists the variants that this Pipe supports.
- For each supported variant, implement its `_to_<VARIANT>` method, which returns a new Variant for the Pipe.
- Implement the Variant itself. To do so, create a new class that inherits from the appropriate Variant (e.g., `InProcessPipeVariant`) and implement the actual processing logic.

Specifically, each `PipeVariant` should implement an `_iter_impl()` method, which is a generator that processes an individual sample. `next()` is automatically called on this generator by the Variant durint execution.

For reference, `cedar/cedar/pipes/noop.py` provides a skeleton that you can extend to create a new `Pipe`.

### Sources
A `Source` that provides a mechanism to iterate over specific data sources. A Source should instead extend the `Source` class (cedar/cedar/sources/source.py).

A new `Source` should implement a `to_pipe` method, which returns a `Pipe` (potentially with multiple `Variants`) that iterates over the raw source data.

For example, an `IterSourcePipe` wraps an iterator as a Source (cedar/cedar/sources/iterable.py).

### Using cedar: Features and DataSets
Hopefully most `Pipes` and `Sources` will be already provided, so you won't have to build your own. You can simply use the ones provided to build a `Feature` to define your pipeline and then create a `DataSet` to process samples.

A Feature represents a data preprocessing pipeline. To create a feature, create a subclass of `Feature`.
To implement the actual logic, you must define a `_compose(self, source_pipes: List[Pipe])` function.
Specifically, `source_pipes` contains a list of `Sources` which provide raw data (if you have only one source, the input will be at `source_pipes[0]`).

Within `_compose`, functionally chain together multiple `Pipes` to define the dataflow graph, and *return the last pipe*.

To use a `Feature`, you need to do the following.
First, create an object for the feature itself, as well as a `Source` object over your specific dataset.
You will also need to create an `CedarContext`, which contains context needed to connect to specific backends (e.g., Ray). You can create an empty `CedarContext` if you're just performing processing locally.

Secondly, you need to call `feature.apply(source)`, which binds the `source` to the `feature`.

Finally, pass in the context and feature (as a dict mapping feature name to feature object) to create a `DataSet`.
While the DataSet has a lot of different arguments, you don't need to worry about any of them initially, you can just simply iterate over the DataSet object to get running!
By default no optimizations are enabled, and this will execute everything as an `InProcess` Variant.

For example, `cedar/evaluation/pipelines/wikitext103/cedar_dataset.py` contains an example to get you started.

### Optimizer
Once your DataSet is running, the next step is to start applying optimizations.

The DataSet contains an Optimizer, which can be enabled using arguments.

The recommended way to use the optimizer is as follows.

First, define which OptimizerOptions (see `cedar/cedar/compose/optimizer.py`) you want to use. You can turn on/off the following:

- enable_prefetch: Turn on prefetching
- est_throughput: Estimated training throughput in samples/s, leave empty if you want to maximize cedar processing throughput.
- available_local_cpus: The number of local CPU cores available to use.
- enable_offload: Turn this on ONLY if you have the Ray cluster configured (and defined in the CedarContext). This enables cedar to offload processing to the Ray backend.
- enable_reorder: Turn this on to enable reordering
- enable_local_parallelism: Turn this on to enable cedar to use multiple local Drivers to perform processing.
- enable_fusion: Turn this on to enable cedar to perform operator fusion

Next, first run a profiling job by simply setting `run_profiling` to True, and creating the DataSet. For example

```python
ds = DataSet(
    CedarContext(),
    {"feature": feature},
    optimizer_options=OptimizerOptions(
        ...
    ),
    run_profiling=True
)
```

This will automatically run a profiling job and immediately exit. After profiling completes, *this will output a statistics file at `/tmp/<FEATURE_NAME>_profile.yml`*. Move this file to a safe place to use for the next step!

We'll next use the profiling results to run the Optimizer to do this, create a new DataSet, and simply turn on the generate_plan flat to True. This will output an optimized plan and immediately exit. After this completes, this will create a plan config file at `/tmp/cedar_<FEATURE_NAME>_plan.yml`. Save this file for the next step.

```python
ds = DataSet(
    CedarContext(),
    {"feature": feature},
    optimizer_options=OptimizerOptions(
        ...
    ),
    generate_plan=True
)
```

Finally, let's run the actual processing with the optimized plan.
To do this, simply pass in the config file generated in the last step to the DataSet.
The DataSet will automatically configure the Feature according to this plan.

```python
ds = DataSet(
    CedarContext(),
    {"feature": feature},
    feature_config="<PATH_TO_PLAN>.yml",
)

for batch in ds:
    ...
```

Note that you can also simply create the DataSet with `enable_optimizer=True`, and the DataSet will automatically run all of the above steps. However, breaking this down step by step allows you to understand/repeat what is happening under the hood.

```python
ds = DataSet(
    CedarContext(),
    {"feature": feature},
    enable_optimizer=True,
)

for batch in ds:
    ...
```

### Controller/Auto-Tuning
The Controller performs auto-tuning of Variants in order to meet a training throughput demand. To enable the controller, set the `enable_controller` flag to `True` when creating the DataSet.

### Running with eval_cedar.py
Finally, `evaluation/eval_cedar.py` provides a utilities to automatically create the DataSet, run profiling, generate the plan, execute the pipeline, and report results. Here is an end-to-end example of how you would use this.

For example let's use `cedar/evaluation/pipelines/wikitext103/cedar_dataset.py` as our example.

First, create a new Python file that will define your dataset. In that file, create a new Feature. Then, create a function defined as `get_dataset(spec: CedarEvalSpec) -> DataSet`. In this function, instantiate your Feature, Source, and create a DataSet.
Hook up the options from the `spec` to the appropriate options in the `DataSet`. 

```python
    if spec.config:
        dataset = DataSet(
            ctx,
            {"feature": feature},
            spec.config,
            enable_controller=False,
            enable_optimizer=False,
        )
    else:
        dataset = DataSet(
            ctx,
            {"feature": feature},
            enable_controller=not spec.disable_controller,
            enable_optimizer=not spec.disable_optimizer,
            profiled_data=spec.profiled_stats,
            run_profiling=spec.run_profiling,
            optimizer_options=OptimizerOptions(
                enable_prefetch=not spec.disable_prefetch,
                est_throughput=None,
                available_local_cpus=mp.cpu_count(),
                enable_offload=not spec.disable_offload,
                enable_reorder=not spec.disable_reorder,
                enable_local_parallelism=not spec.disable_parallelism,
                enable_fusion=not spec.disable_fusion,
            ),
            generate_plan=spec.generate_plan,
        )
```

Return the created dataset from the function.

Next, simply write a main() function in the same file that tries to read a few samples from your DataSet. This ensures that your Feature is working as intended.

```python
    ds = get_dataset(CedarEvalSpec(1, None, 1))
    for s in ds:
        print(s)
        break
```

```bash
python cedar_dataset.py
# Check that reasonable data is printed!
```


Once you've validated that your feature works, you can next begin using `eval_cedar.py` to perform the steps discussed in the [Optimizer](#optimizer) section above.

Run a profiling step to generate profile stats. Be sure to save those stats.

```bash
taskset -c 0 python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_dataset.py --run_profiling

# Save the stats file for later use if you want!
mv /tmp/feature_profile.yaml <PATH_TO_STATS>
```

Next, let's use those stats to generate the optimized plan. In this case, we won't be using Ray, so let's disable the offloading functionality.

```bash
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_dataset.py --profiled_stats /tmp/feature_profile.yml --generate_plan --disable_offload

# Save the plan for later use if you want!
mv /tmp/cedar_optimized_plan.yml <PATH_TO_PLAN>
```
Take a look at the optimized plan to understand what the Optimizer did!

Finally, when you are ready to run training, you can do so by passing in the plan.

```bash
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_dataset.py --master_feature_config /tmp/cedar_optimized_plan.yml --num_total_samples 100000
```

## Running Paper Experiments

Refer to the README in `evaluation` for further instructions on how to run the evaluation scripts.
The unified ten-workload comparison implementations for PyTorch, tf.data, Ray
Data, Data-Juicer, Plumber, FastFlow, and Pecan are documented in
`evaluation/baselines/README.md`.

## Troubleshooting
* Connecting to the head node: Make sure that your firewall rules allows incoming connections to
the head node (likely at port 10001).

* Permission issues performing `ray up`: Make sure your VM has necessary API permissions.

## Tests
To run tests, simply use pytest: `pytest tests/`
