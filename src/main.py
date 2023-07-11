import os
import json
import time
import hydra
import logging
import numpy as np
import tensorflow as tf
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path
from models import ModelBuilder
from utils import styled_print, split_data
from trainer import get_classification_loss_fn, train_and_evaluate_model
from datapipeline import DatasetBuilder, DatapipelineBuilder

log = logging.getLogger(__name__)


@hydra.main(
    config_path="../configs", config_name="sample", version_base="1.2"
)

def main(cfg: DictConfig) -> None:
    # Print config
    if cfg.experiment_config.print_config:
        styled_print(f"Hydra Config for {HydraConfig.get().job.name}", header=True)
        styled_print(OmegaConf.to_yaml(cfg))
    
    styled_print(f"Current working directory : {os.getcwd()}", header=True)
    styled_print(f"Orig working directory    : {get_original_cwd()}", header=True)

    # Prepare the dataset.
    batch_size = cfg.training_config.batch_size
    dataset_builder = DatasetBuilder(
        dataset_id=cfg.data_config.dataset_id, flatten=cfg.data_config.flatten
    )
    (x_train, y_train), (x_test, y_test) = dataset_builder.get_dataset()
    (x_train, y_train), (x_val, y_val) = split_data(
        x_train, y_train, cfg.training_config.training_observations_cnt
    )

    train_dg = DatapipelineBuilder(x_train, y_train).create_datapipeline(
        cfg.training_config.batch_size
    )
    val_dg = DatapipelineBuilder(x_val, y_val).create_datapipeline(
        cfg.training_config.batch_size
    )
    test_dg = DatapipelineBuilder(x_test, y_test).create_datapipeline(
        cfg.training_config.batch_size
    )

    epochs = cfg.training_config.epochs
    repetitions_cnt = cfg.experiment_config.repetitions_cnt

    # overall stats
    overall_stats = []
    for repetition_ind in range(repetitions_cnt):
        log.info(f"Repetition #{repetition_ind} for {HydraConfig.get().job.name}")
        # reset state after every iteration, reduces memory consumption
        tf.keras.backend.clear_session()
        # Re-init the model and associated objects
        model_builder = ModelBuilder(
            model_id=cfg.model_config.model_id,
            input_shape=cfg.model_config.input_shape,
            output_shape=cfg.model_config.output_shape,
            output_activation=cfg.model_config.output_activation,
            model_config={
                "kernel_initializer": cfg.model_config.kernel_initializer,
                "bias_initializer": cfg.model_config.bias_initializer,
            },
        )
        model = model_builder.get_model()

        # Instantiate an optimizer to train the model.
        optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

        # Instantiate a loss function.
        loss_fn = get_classification_loss_fn(
            loss_id=cfg.training_config.loss.loss_id,
            logits=cfg.training_config.loss.loss_logits,
        )

        # Prepare the metrics.
        train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        # Run experiment
        stats = train_and_evaluate_model(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_acc_metric=train_acc_metric,
            val_acc_metric=val_acc_metric,
            test_acc_metric=test_acc_metric,
            batch_size=batch_size,
            train_dg=train_dg,
            val_dg=val_dg,
            test_dg=test_dg,
            epochs=epochs,
            initializer=cfg.model_config.kernel_initializer,
            log=log,
        )

        log.info(f"Summary: {stats}")
        overall_stats.append(stats)

    with open(cfg.experiment_config.stats_file_name, "w") as fp:
        json.dump(overall_stats, fp, indent=4, sort_keys=True)

if __name__ == "__main__":
    main()