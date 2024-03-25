"""Configuration module to setup nucml's working environment."""

import yaml
from pathlib import Path

config_yaml = Path(__file__).parents[0] / 'config.yaml'


def _get_config():
    with open(config_yaml) as f:
        config = yaml.load(f)
    return config


def _save_config(config):
    with open(config_yaml, 'w') as f:
        yaml.dump(config, f)


def set_data_paths(new_values_dict):
    config = _get_config()

    for key, value in new_values_dict.items():
        config['DATA_PATHS'][key] = value

    _save_config(config)


def _set_path(key, value):
    config = _get_config()
    config[key] = value
    _save_config(config)


def set_serpent_path(path):
    _set_path('SERPENT_PATH', path)


def set_matlab_path(path):
    _set_path('MATLAB_PATH', path)


def set_benchmarking_path(path):
    _set_path('BENCHMARKING_TEMPLATE_PATH', path)
