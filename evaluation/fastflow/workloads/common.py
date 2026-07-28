"""Shared FastFlow application for serializable OptimalCedar tf.data graphs."""

from __future__ import annotations

import importlib
from pathlib import Path

import fastflow as ff
import tensorflow as tf

from evaluation.fastflow.examples.eval_app_runner import App
from evaluation.tf_utils import TFEvalSpec


class BaselineFastFlowModel(ff.FastFlowModel):
    def call(self, inputs):
        terms = []
        for tensor in tf.nest.flatten(inputs):
            if tensor.dtype.is_floating or tensor.dtype.is_integer:
                tensor = tf.cast(tensor, tf.float32)
                terms.append(
                    tf.reduce_mean(tensor, axis=tf.range(1, tf.rank(tensor)))
                )
        if not terms:
            raise TypeError("FastFlow workload produced no numeric tensor.")
        return tf.add_n(terms)

    def __deepcopy__(self, memo=None):
        return BaselineFastFlowModel()


class BaselineFastFlowApp(App):
    workload = None

    def create_model(self):
        model = BaselineFastFlowModel()
        model.compile(optimizer="sgd", loss=self._dummy_loss)
        return model

    @staticmethod
    def _dummy_loss(y_true, y_pred):
        # The scalar reduction forces the complete dataset element to be
        # consumed while adding negligible model-side work.
        return tf.reduce_mean(tf.square(y_pred - tf.cast(y_true, tf.float32)))

    def _as_training_pair(self, *values):
        inputs = values[0] if len(values) == 1 else values
        first = tf.nest.flatten(inputs)[0]
        target = tf.zeros([tf.shape(first)[0]], dtype=tf.float32)
        return inputs, target

    def create_dataset(self, num_parallel):
        if self.workload == "coco":
            from evaluation.fastflow.workloads.coco_pipeline import (
                build_dataset,
            )

            dataset = build_dataset(
                Path(self.args.data_prefix),
                num_parallel,
                self.args.batch,
            )
        else:
            module = importlib.import_module(
                f"evaluation.pipelines.{self.workload}.tf_dataset"
            )
            kwargs = {"fastflow": True}
            if self.args.data_prefix:
                kwargs["dataset_path"] = self.args.data_prefix
            dataset = module.get_dataset(
                TFEvalSpec(
                    batch_size=self.args.batch,
                    num_parallel_calls=num_parallel,
                    num_total_samples=self.args.num_samples,
                    kwargs=kwargs,
                )
            )
            if self.workload in {"commonvoice", "wikitext103"}:
                dataset = dataset.batch(self.args.batch)

        return dataset.map(
            self._as_training_pair,
            num_parallel_calls=num_parallel,
            name="fastflow_training_pair",
        )

    def create_valid_dataset(self, num_parallel):
        return None


__all__ = ["BaselineFastFlowApp", "BaselineFastFlowModel"]
