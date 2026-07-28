from cedar.config import CedarContext
from cedar.pipes import PipeVariantType
from cedar.sources import COCOSource, IterSource, LocalFSSource, LocalLineSource
from .utils import (
    check_fault_tolerance_funcs,
    check_replay_files_image_folder,
    check_line_source_replay_output,
)

import pathlib
import json

import pytest
from PIL import Image


def test_basic_itersource():
    data = [1, 2, 3, 4, 5]
    source = IterSource(data)
    pipe = source.to_pipe()

    pipe.mutate(CedarContext(), PipeVariantType.INPROCESS)
    assert pipe.pipe_variant is not None

    out = []

    # Test results
    for x in pipe.pipe_variant:
        out.append(x.data)
    assert out == data
    assert pipe.pipe_variant.num_yielded == len(data)

    # Check fault tolerance basic funcs
    check_fault_tolerance_funcs(pipe, 1000, 1, 0, 0, 4)


def test_coco_source_selects_split_and_keeps_images_without_annotations(tmp_path):
    image_dir = tmp_path / "train2017"
    annotation_dir = tmp_path / "annotations"
    image_dir.mkdir()
    annotation_dir.mkdir()
    images = []
    for image_id in (1, 2):
        file_name = f"{image_id:012d}.jpg"
        Image.new("RGB", (4, 3)).save(image_dir / file_name)
        images.append(
            {"id": image_id, "file_name": file_name, "width": 4, "height": 3}
        )
    payload = {
        "images": images,
        "annotations": [
            {"image_id": 1, "bbox": [0, 0, 2, 2], "category_id": 7}
        ],
    }
    (annotation_dir / "instances_train2017.json").write_text(
        json.dumps(payload)
    )

    pipe = COCOSource(str(tmp_path), split="train2017").to_pipe()
    pipe.mutate(CedarContext(), PipeVariantType.INPROCESS)
    samples = [sample.data for sample in pipe.pipe_variant]

    assert len(samples) == 2
    assert samples[0]["boxes"].shape == (1, 4)
    assert samples[1]["boxes"].shape == (0, 4)


def test_localfssource_file(tmp_path):
    file1_path = tmp_path / "file1.txt"
    file2_path = tmp_path / "file2.txt"
    file1_path.touch()
    file2_path.touch()

    source = LocalFSSource(str(file1_path))
    pipe = source.to_pipe()
    pipe.mutate(CedarContext(), PipeVariantType.INPROCESS)

    # Test results
    expected = [str(file1_path)]
    out = []
    for x in pipe.pipe_variant:
        out.append(x.data)

    assert out == expected

    source = LocalFSSource(str(tmp_path))
    pipe = source.to_pipe()
    pipe.mutate(CedarContext(), PipeVariantType.INPROCESS)

    expected = [str(file1_path), str(file2_path)]
    out = []
    for x in pipe.pipe_variant:
        out.append(x.data)

    assert out == expected
    assert pipe.pipe_variant.num_yielded == len(expected)

    # Check fault tolerance basic funcs
    check_fault_tolerance_funcs(pipe, 1000, 1, 0, 0, 1)


def test_localfssource_max_samples_limits_a_deterministic_prefix(tmp_path):
    for i in range(5):
        (tmp_path / f"file{i}.txt").touch()

    pipe = LocalFSSource(str(tmp_path), max_samples=3).to_pipe()
    pipe.mutate(CedarContext(), PipeVariantType.INPROCESS)

    assert [sample.data for sample in pipe.pipe_variant] == [
        str(tmp_path / f"file{i}.txt") for i in range(3)
    ]


def test_locallinepipe():
    test_dir = pathlib.Path(__file__).resolve().parents[0]
    test_file = pathlib.Path(test_dir) / "data/test_text.txt"
    source = LocalLineSource(str(test_file))

    pipe = source.to_pipe()
    pipe.mutate(CedarContext(), PipeVariantType.INPROCESS)

    # Test line read
    expected = ["hello", "world"]
    out = []
    for x in pipe.pipe_variant:
        out.append(x.data)

    assert out == expected
    assert pipe.pipe_variant.num_yielded == len(expected)

    # Check fault tolerance funcs
    check_fault_tolerance_funcs(pipe, 1000, 1, 0, 0, 1)


def test_locallinesource_max_samples_limits_a_deterministic_prefix(tmp_path):
    path = tmp_path / "lines.txt"
    path.write_text("zero\none\ntwo\nthree\n")

    pipe = LocalLineSource(str(path), max_samples=2).to_pipe()
    pipe.mutate(CedarContext(), PipeVariantType.INPROCESS)

    assert [sample.data for sample in pipe.pipe_variant] == ["zero", "one"]
    assert pipe.pipe_variant.length == 2


@pytest.mark.parametrize(
    "fully_iterated_over_iterable_source_pipe", [(0, 1000)], indirect=True
)
def test_empty_input(fully_iterated_over_iterable_source_pipe):
    """
    Test handling of empty input.
    """
    (
        pipe,
        num_samples,
        num_samples_per_partition,
        total_in_flight_partitions,
        total_fully_sent_partitions,
        first_sampled_id,
        last_sample_id,
    ) = fully_iterated_over_iterable_source_pipe

    check_fault_tolerance_funcs(
        pipe,
        num_samples_per_partition,
        total_in_flight_partitions,
        total_fully_sent_partitions,
        first_sampled_id,
        last_sample_id,
    )


@pytest.mark.parametrize(
    "fully_iterated_over_iterable_source_pipe", [(1, 1000)], indirect=True
)
def test_one_sized_input(fully_iterated_over_iterable_source_pipe):
    """
    Test handling of input of size 1.
    """
    (
        pipe,
        num_samples,
        num_samples_per_partition,
        total_in_flight_partitions,
        total_fully_sent_partitions,
        first_sampled_id,
        last_sample_id,
    ) = fully_iterated_over_iterable_source_pipe

    check_fault_tolerance_funcs(
        pipe,
        num_samples_per_partition,
        total_in_flight_partitions,
        total_fully_sent_partitions,
        first_sampled_id,
        last_sample_id,
    )


@pytest.mark.parametrize(
    "fully_iterated_over_iterable_source_pipe", [(1000, 1000)], indirect=True
)
def test_full_partition(fully_iterated_over_iterable_source_pipe):
    """
    Tests handling of one partition that has exactly partition_size.
    """
    (
        pipe,
        num_samples,
        num_samples_per_partition,
        total_in_flight_partitions,
        total_fully_sent_partitions,
        first_sampled_id,
        last_sample_id,
    ) = fully_iterated_over_iterable_source_pipe

    check_fault_tolerance_funcs(
        pipe,
        num_samples_per_partition,
        total_in_flight_partitions,
        total_fully_sent_partitions,
        first_sampled_id,
        last_sample_id,
    )


@pytest.mark.parametrize(
    "fully_iterated_over_iterable_source_pipe", [(1500, 1000)], indirect=True
)
def test_two_partitions_non_full(fully_iterated_over_iterable_source_pipe):
    """
    Tests handling of two partitions, one of which has partition_size
    with the other having less samples than partition_size.
    """
    (
        pipe,
        num_samples,
        num_samples_per_partition,
        total_in_flight_partitions,
        total_fully_sent_partitions,
        first_sampled_id,
        last_sample_id,
    ) = fully_iterated_over_iterable_source_pipe

    check_fault_tolerance_funcs(
        pipe,
        num_samples_per_partition,
        total_in_flight_partitions,
        total_fully_sent_partitions,
        first_sampled_id,
        last_sample_id,
    )


@pytest.mark.parametrize(
    "fully_iterated_over_iterable_source_pipe", [(2000, 1000)], indirect=True
)
def test_two_partitions_full(fully_iterated_over_iterable_source_pipe):
    """
    Tests handling of two partitions, both of which have size
    partition_size.
    """
    (
        pipe,
        num_samples,
        num_samples_per_partition,
        total_in_flight_partitions,
        total_fully_sent_partitions,
        first_sampled_id,
        last_sample_id,
    ) = fully_iterated_over_iterable_source_pipe

    check_fault_tolerance_funcs(
        pipe,
        num_samples_per_partition,
        total_in_flight_partitions,
        total_fully_sent_partitions,
        first_sampled_id,
        last_sample_id,
    )


@pytest.mark.parametrize(
    "fully_iterated_over_iterable_source_pipe", [(5, 5)], indirect=True
)
def test_changing_partition_size_full(
    fully_iterated_over_iterable_source_pipe,
):
    """
    Tests handling of one partition with a changed partition size
    compared to default. Num elements is equal to partition size.
    """
    (
        pipe,
        num_samples,
        num_samples_per_partition,
        total_in_flight_partitions,
        total_fully_sent_partitions,
        first_sampled_id,
        last_sample_id,
    ) = fully_iterated_over_iterable_source_pipe

    check_fault_tolerance_funcs(
        pipe,
        num_samples_per_partition,
        total_in_flight_partitions,
        total_fully_sent_partitions,
        first_sampled_id,
        last_sample_id,
    )


@pytest.mark.parametrize(
    "fully_iterated_over_iterable_source_pipe", [(6, 5)], indirect=True
)
def test_changing_partition_size_non_full(
    fully_iterated_over_iterable_source_pipe,
):
    """
    Tests handling of one partition with a changed partition size
    compared to default. Num elements is not equal to partition size.
    """
    (
        pipe,
        num_samples,
        num_samples_per_partition,
        total_in_flight_partitions,
        total_fully_sent_partitions,
        first_sampled_id,
        last_sample_id,
    ) = fully_iterated_over_iterable_source_pipe

    check_fault_tolerance_funcs(
        pipe,
        num_samples_per_partition,
        total_in_flight_partitions,
        total_fully_sent_partitions,
        first_sampled_id,
        last_sample_id,
    )


@pytest.mark.parametrize(
    "fully_iterated_over_iterable_source_pipe", [(1, 1)], indirect=True
)
def test_partition_size_one_full(fully_iterated_over_iterable_source_pipe):
    """
    Tests handling of one partition with a partition size of 1.
    Num elements is equal to partition size.
    """
    (
        pipe,
        num_samples,
        num_samples_per_partition,
        total_in_flight_partitions,
        total_fully_sent_partitions,
        first_sampled_id,
        last_sample_id,
    ) = fully_iterated_over_iterable_source_pipe

    check_fault_tolerance_funcs(
        pipe,
        num_samples_per_partition,
        total_in_flight_partitions,
        total_fully_sent_partitions,
        first_sampled_id,
        last_sample_id,
    )


@pytest.mark.parametrize(
    "fully_iterated_over_iterable_source_pipe", [(0, 1)], indirect=True
)
def test_partition_size_one_non_full(fully_iterated_over_iterable_source_pipe):
    """
    Tests handling of one partition with a partition size of 1.
    Num elements is not equal to partition size.
    """
    (
        pipe,
        num_samples,
        num_samples_per_partition,
        total_in_flight_partitions,
        total_fully_sent_partitions,
        first_sampled_id,
        last_sample_id,
    ) = fully_iterated_over_iterable_source_pipe

    check_fault_tolerance_funcs(
        pipe,
        num_samples_per_partition,
        total_in_flight_partitions,
        total_fully_sent_partitions,
        first_sampled_id,
        last_sample_id,
    )


@pytest.mark.parametrize(
    "fully_iterated_over_iterable_source_pipe", [(10001, 2)], indirect=True
)
def test_many_partitions_non_full(fully_iterated_over_iterable_source_pipe):
    """
    Tests handling of many partitions.
    Last partition needs to be sealed explictily.
    """
    (
        pipe,
        num_samples,
        num_samples_per_partition,
        total_in_flight_partitions,
        total_fully_sent_partitions,
        first_sampled_id,
        last_sample_id,
    ) = fully_iterated_over_iterable_source_pipe

    check_fault_tolerance_funcs(
        pipe,
        num_samples_per_partition,
        total_in_flight_partitions,
        total_fully_sent_partitions,
        first_sampled_id,
        last_sample_id,
    )


@pytest.mark.parametrize(
    "fully_iterated_over_iterable_source_pipe", [(10000, 2)], indirect=True
)
def test_many_partitions_full(fully_iterated_over_iterable_source_pipe):
    """
    Tests handling of many partitions.
    Last partition does not need to be sealed explictily.
    """
    (
        pipe,
        num_samples,
        num_samples_per_partition,
        total_in_flight_partitions,
        total_fully_sent_partitions,
        first_sampled_id,
        last_sample_id,
    ) = fully_iterated_over_iterable_source_pipe

    check_fault_tolerance_funcs(
        pipe,
        num_samples_per_partition,
        total_in_flight_partitions,
        total_fully_sent_partitions,
        first_sampled_id,
        last_sample_id,
    )


@pytest.mark.parametrize(
    "noop_dataset_with_variable_input", [10000], indirect=True
)
def test_replay_itersource_basic(noop_dataset_with_variable_input):
    outputs = []
    counter = 0
    for x in noop_dataset_with_variable_input:
        outputs.append(x)
        # Simulate losing all samples
        if counter == 9999:
            source_pipes = noop_dataset_with_variable_input._get_source_pipes()
            feature_names = (
                noop_dataset_with_variable_input._get_feature_names()
            )
            variant = source_pipes[feature_names[0]][0].pipe_variant
            variant.fully_sent_partitions = variant.sealed_partitions
            variant.sealed_partitions = {}
            ds_iter = noop_dataset_with_variable_input.dataset_iter
            ds_iter.partitions_received = {}
            variant.enable_replay()
        counter += 1

    expected_outputs = [i for i in range(10000)]
    expected_outputs.extend([i for i in range(10000)])

    assert outputs == expected_outputs


@pytest.mark.parametrize(
    "noop_dataset_with_variable_input", [10000], indirect=True
)
def test_replay_itersource_mid_dataset(noop_dataset_with_variable_input):
    outputs = []
    counter = 0
    for x in noop_dataset_with_variable_input:
        outputs.append(x)
        # Simulate some samples
        if counter == 4998:
            source_pipes = noop_dataset_with_variable_input._get_source_pipes()
            feature_names = (
                noop_dataset_with_variable_input._get_feature_names()
            )
            variant = source_pipes[feature_names[0]][0].pipe_variant
            variant.fully_sent_partitions = variant.sealed_partitions
            variant.sealed_partitions = {}
            ds_iter = noop_dataset_with_variable_input.dataset_iter
            ds_iter.partitions_received = {}
            variant.enable_replay()
        counter += 1

    expected_outputs = [i for i in range(4999)]
    expected_outputs.extend([i for i in range(4999)])
    expected_outputs.extend([i for i in range(4999, 10000)])

    assert outputs == expected_outputs


@pytest.mark.parametrize(
    "noop_dataset_with_variable_input", [10000], indirect=True
)
def test_replay_itersource_no_samples_lost_full_dataset(
    noop_dataset_with_variable_input,
):
    outputs = []
    counter = 0
    for x in noop_dataset_with_variable_input:
        outputs.append(x)
        # Simulate some samples
        if counter == 9999:
            source_pipes = noop_dataset_with_variable_input._get_source_pipes()
            feature_names = (
                noop_dataset_with_variable_input._get_feature_names()
            )
            variant = source_pipes[feature_names[0]][0].pipe_variant
            variant.enable_replay()
        counter += 1

    expected_outputs = [i for i in range(10000)]

    assert outputs == expected_outputs


@pytest.mark.parametrize(
    "noop_dataset_with_variable_input", [10000], indirect=True
)
def test_replay_itersource_only_in_flight_samples_lost_mid_dataset(
    noop_dataset_with_variable_input,
):
    outputs = []
    counter = 0
    for x in noop_dataset_with_variable_input:
        outputs.append(x)
        # Simulate some samples
        if counter == 4998:
            source_pipes = noop_dataset_with_variable_input._get_source_pipes()
            feature_names = (
                noop_dataset_with_variable_input._get_feature_names()
            )
            variant = source_pipes[feature_names[0]][0].pipe_variant
            variant.enable_replay()
            ds_iter = noop_dataset_with_variable_input.dataset_iter
            ds_iter.partitions_received.pop(4)
        counter += 1

    expected_outputs = [i for i in range(0, 4999)]
    expected_outputs.extend([i for i in range(4000, 4999)])
    expected_outputs.extend([i for i in range(4999, 10000)])

    assert outputs == expected_outputs


def test_replay_file_lister_source_full_dataset(noop_dataset_with_file_source):
    dataset, variant = noop_dataset_with_file_source

    counter = 0
    started_replay = False
    before_replay = []
    outputs = []
    for x in dataset:
        outputs.append(x)
        if not started_replay:
            before_replay.append(x)

        # Simulate all samples
        if counter == 9:
            variant.fully_sent_partitions = variant.sealed_partitions
            variant.sealed_partitions = {}
            dataset.dataset_iter.partitions_received = {}
            variant.enable_replay()
            started_replay = True
        counter += 1

    check_replay_files_image_folder(outputs, before_replay)


def test_replay_file_lister_source_mid_dataset(noop_dataset_with_file_source):
    dataset, variant = noop_dataset_with_file_source

    counter = 0
    started_replay = False
    before_replay = []
    outputs = []
    for x in dataset:
        outputs.append(x)
        if not started_replay:
            before_replay.append(x)

        # Simulate some samples lost
        if counter == 6:
            variant.fully_sent_partitions = variant.sealed_partitions
            variant.sealed_partitions = {}
            dataset.dataset_iter.partitions_received = {}
            variant.enable_replay()
            started_replay = True
        counter += 1

    check_replay_files_image_folder(outputs, before_replay)


def test_replay_file_lister_source_no_data_lost(noop_dataset_with_file_source):
    dataset, variant = noop_dataset_with_file_source

    counter = 0
    started_replay = False
    before_replay = []
    outputs = []
    for x in dataset:
        outputs.append(x)
        if not started_replay:
            before_replay.append(x)

        if counter == 1:
            variant.enable_replay()

        if counter == 5:
            variant.enable_replay()

        if counter == 9:
            variant.enable_replay()

        counter += 1

    check_replay_files_image_folder(outputs, before_replay, 2, False, True)


def test_replay_file_lister_source_only_in_flight_samples_lost_mid_dataset(
    noop_dataset_with_file_source,
):
    dataset, variant = noop_dataset_with_file_source

    counter = 0
    started_replay = False
    before_replay = []
    outputs = []
    for x in dataset:
        outputs.append(x)
        if not started_replay:
            before_replay.append(x)

        # Simulate some samples lost
        if counter == 6:
            variant.enable_replay()
            started_replay = True
            dataset.dataset_iter.partitions_received.pop(3)
        counter += 1

    check_replay_files_image_folder(outputs, before_replay, 2, True)


def test_replay_line_reader_source_full_dataset(
    noop_dataset_with_line_reader_source,
):
    dataset, variant = noop_dataset_with_line_reader_source

    counter = 0
    outputs = []
    for x in dataset:
        outputs.append(x)

        # Simulate losing all samples
        if counter == 6:
            variant.fully_sent_partitions = variant.sealed_partitions
            variant.sealed_partitions = {}
            dataset.dataset_iter.partitions_received = {}
            variant.enable_replay()
        counter += 1

    replay_line_start = 0
    replay_line_end = 6
    check_line_source_replay_output(
        replay_line_start, replay_line_end, outputs
    )


def test_replay_line_reader_source_mid_lost(
    noop_dataset_with_line_reader_source,
):
    dataset, variant = noop_dataset_with_line_reader_source

    counter = 0
    outputs = []
    for x in dataset:
        outputs.append(x)

        # Simulate losing some samples
        if counter == 4:
            variant.fully_sent_partitions = variant.sealed_partitions
            variant.sealed_partitions = {}
            dataset.dataset_iter.partitions_received = {}
            variant.enable_replay()
        counter += 1

    replay_line_start = 0
    replay_line_end = 4
    check_line_source_replay_output(
        replay_line_start, replay_line_end, outputs
    )


def test_replay_line_reader_source_last_in_flight_lost(
    noop_dataset_with_line_reader_source,
):
    dataset, variant = noop_dataset_with_line_reader_source

    counter = 0
    outputs = []
    for x in dataset:
        outputs.append(x)

        # Activate replay with last in-flight lost
        if counter == 6:
            variant.enable_replay()
            dataset.dataset_iter.partitions_received.pop(3)
        counter += 1

    replay_line_start = 6
    replay_line_end = 6
    check_line_source_replay_output(
        replay_line_start, replay_line_end, outputs
    )


def test_replay_line_reader_source_mid_in_flight_lost(
    noop_dataset_with_line_reader_source,
):
    dataset, variant = noop_dataset_with_line_reader_source

    counter = 0
    outputs = []
    for x in dataset:
        outputs.append(x)

        # Activate replay with last in-flight lost
        if counter == 4:
            variant.enable_replay()
            dataset.dataset_iter.partitions_received.pop(2)
        counter += 1

    replay_line_start = 4
    replay_line_end = 4
    check_line_source_replay_output(
        replay_line_start, replay_line_end, outputs
    )


def test_replay_line_reader_source_enable_replay_but_nothing_to_replay(
    noop_dataset_with_line_reader_source,
):
    dataset, variant = noop_dataset_with_line_reader_source

    counter = 0
    outputs = []
    for x in dataset:
        outputs.append(x)

        # Activate replay with last in-flight lost
        if counter == 3:
            variant.enable_replay()

        if counter == 5:
            variant.enable_replay()

        counter += 1

    replay_line_start = None
    replay_line_end = None
    check_line_source_replay_output(
        replay_line_start, replay_line_end, outputs
    )


def test_localfssource_sharded(tmp_path):
    files = []
    for i in range(10):
        file_name = "file_" + str(i) + ".txt"
        file_path = tmp_path / file_name
        file_path.touch()
        files.append(file_path)

    for i in range(3):
        source = LocalFSSource(files, rank_spec=(3, i))
        pipe = source.to_pipe()
        pipe.mutate(CedarContext(), PipeVariantType.INPROCESS)

        out = []
        for x in pipe.pipe_variant:
            out.append(x.data)

        assert out == [x for idx, x in enumerate(files) if (idx - i) % 3 == 0]


def test_linereader_sharded():
    test_dir = pathlib.Path(__file__).resolve().parents[0]
    test_file = pathlib.Path(test_dir) / "data/test_text_3.txt"

    for i in range(3):
        source = LocalLineSource(str(test_file), rank_spec=(3, i))
        pipe = source.to_pipe()
        pipe.mutate(CedarContext(), PipeVariantType.INPROCESS)

        out = []
        for x in pipe.pipe_variant:
            out.append(x.data)

        assert len(out) == 4
