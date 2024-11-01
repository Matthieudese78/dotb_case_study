#!/usr/bin/python3
from __future__ import annotations

import argparse
import os

import yaml


def get_config_file(default_config_path, default_input):
    """
    Parses the argument for a .yaml input file

    Parameters : default_config_path
    For now = the root directory
    - appends the root dir name to the argument
    - if the .yaml exists, returns the input.file path
    - if not : uses ballistic_default_input.yaml file as input
    """
    # Define the default config file path
    # default_config_path = os.path.join(os.path.dirname(__file__), 'default_config.yaml')
    default_config_path = f'{default_config_path}/{default_input}'

    # Create an ArgumentParser instance
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument(
        'config',
        nargs='?',
        default=default_config_path,
        help='Path to the config file (optional)',
    )

    # Parse the arguments
    args = parser.parse_args()

    # Get the config file path
    config_path = args.config

    # Check if the provided config file exists
    if not os.path.exists(config_path):
        print(
            f"Warning: Config file '{config_path}' does not exist. Using default config.",
        )
        config_path = default_config_path

    return config_path


# turns the entries of a config.yaml file into a dict :


def load_config(file_path):
    """
    Turns the entries of a config.yaml file into a dict.
    """
    with open(file_path) as config_file:
        return yaml.safe_load(config_file)
