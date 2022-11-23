import tensorflow as tf
import tensorflow_ranking as tfr
from typing import Optional

class ZlibDatasetBuilder(tfr.keras.pipeline.SimpleDatasetBuilder):
    def _build_dataset(self,
                     file_pattern: str,
                     batch_size: int,
                     list_size: Optional[int] = None,
                     randomize_input: bool = True,
                     num_epochs: Optional[int] = None) -> tf.data.Dataset:
        dataset = tfr.data.build_ranking_dataset(
        file_pattern=file_pattern,
        data_format=tfr.data.ELWC,
        batch_size=batch_size,
        list_size=list_size,
        context_feature_spec=dict(
            list(self._context_feature_spec.items()) +
            list(self._training_only_context_spec.items())),
        example_feature_spec=dict(
            list(self._example_feature_spec.items()) +
            list(self._training_only_example_spec.items())),
        mask_feature_name=self._mask_feature_name,
        reader=self._hparams.dataset_reader,
        reader_args=['ZLIB'],
        num_epochs=num_epochs,
        shuffle=randomize_input,
        shuffle_buffer_size=1000,
        shuffle_seed=None,
        prefetch_buffer_size=10000,
        reader_num_threads=64,
        sloppy_ordering=True,
        drop_final_batch=False,
        shuffle_examples=False)

        return dataset.map(
        self._features_and_labels,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)