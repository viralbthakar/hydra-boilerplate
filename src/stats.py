"""
1. Read all JSON files in subdirectory of a given directory.
2. Aggregate these JSON files in one data structure
3. Save the resulting aggregated data structure in a new JSON file.
"""

import json
import os
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path

log = logging.getLogger(__name__)


def read_json_files(directory, file_name):
    """
    Read all JSON files in subdirectory of a given directory.
    """
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == file_name:
                json_files.append(os.path.join(root, file))
    return json_files


def aggregate_json_files(json_files):
    """
    Aggregate these JSON files in one data structure
    """
    aggregated_data = []
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
        aggregated_data = [*aggregated_data, *data]
    return aggregated_data


def save_aggregated_data(aggregated_data, output_file):
    """
    Save the resulting aggregated data structure in a new JSON file.
    """
    with open(output_file, 'w') as f:
        json.dump(aggregated_data, f, indent=4, sort_keys=True)


@hydra.main(config_path="../configs", config_name="sample", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main function
    """
    root_dir = to_absolute_path(cfg.stats_aggregation.input_root_dir)
    input_file_name = cfg.stats_aggregation.input_file_name
    output_file_name = cfg.stats_aggregation.output_file_name

    json_files = read_json_files(root_dir, input_file_name)
    log.info(
        f"Found {len(json_files)} {input_file_name} files to process in {root_dir}")
    aggregated_data = aggregate_json_files(json_files)
    log.info(
        f"Saving {len(aggregated_data)} aggregated observations to {output_file_name}")
    save_aggregated_data(aggregated_data, os.path.join(
        root_dir, output_file_name))


if __name__ == '__main__':
    main()