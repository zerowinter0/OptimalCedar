import tensorflow as tf
import tensorflow_addons as tfa
import pathlib

from evaluation.tf_utils import TFEvalSpec

DATASET_LOC = "datasets/imagenette2"
IMG_HEIGHT = 244
IMG_WIDTH = 244
GAUSSIAN_BLUR_KERNEL_SIZE = 11
GCS_PATTERN = "gs://ember-data/imagenette2/train/*/*"


def process_path(img):
    boxes = tf.random.uniform(shape=(1, 4))

    # img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, axis=0)
    img = tf.image.crop_and_resize(img, boxes, [0], [IMG_HEIGHT, IMG_WIDTH])
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
    img = tf.image.random_hue(img, max_delta=0.1)
    img = tf.image.rgb_to_grayscale(img)
    img = tfa.image.gaussian_filter2d(
        img,
        filter_shape=[GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_KERNEL_SIZE],
    )
    img = tf.image.per_image_standardization(img)
    return img


def build_dataset(data_dir, spec):
    if spec.read_from_remote:
        list_of_files = tf.io.gfile.glob(GCS_PATTERN)
        ds = tf.data.Dataset.from_tensor_slices(list_of_files)
    else:
        ds = tf.data.Dataset.list_files(str(data_dir / "*/*"), shuffle=True)

    # ds = tf.data.Dataset.list_files(str(data_dir / "*/*"), shuffle=True)
    map_kwargs = {"num_parallel_calls": spec.num_parallel_calls}
    if spec.kwargs.get("fastflow"):
        map_kwargs["name"] = "prep_begin"
    ds = ds.map(tf.io.read_file, **map_kwargs)
    ds = ds.map(
        lambda x: process_path(x),
        num_parallel_calls=spec.num_parallel_calls,
    )
    ds = ds.batch(spec.batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    if spec.service_addr:
        print(
            "Using tf.data.service with address {}".format(spec.service_addr)
        )
        ds = ds.apply(
            tf.data.experimental.service.distribute(
                processing_mode="distributed_epoch", service=spec.service_addr
            )
        )
    return ds


def get_dataset(spec: TFEvalSpec):
    train_filepath = spec.kwargs.get("dataset_path")
    if not train_filepath:
        data_dir = (
            pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
        )
        train_filepath = pathlib.Path(data_dir) / "imagenette2/train"

    # return gen_files(train_filepath)

    return build_dataset(
        pathlib.Path(train_filepath),
        spec,
    )


if __name__ == "__main__":
    batch_size = 8
    num_workers = tf.data.AUTOTUNE
    data_dir = (
        pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )
    train_filepath = pathlib.Path(data_dir) / "imagenette2/train"

    tf_dataset = get_dataset(TFEvalSpec(1, 1))

    for i, x in enumerate(tf_dataset):
        print(x)
        # print(x.shape)
        print(i)
        break
