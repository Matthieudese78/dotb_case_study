#!/usr/bin/python3
from __future__ import annotations

import argparse
import os

# import yaml
# find


def get_config_file(default_config_path):
    # Define the default config file path
    # default_config_path = os.path.join(os.path.dirname(__file__), 'default_config.yaml')
    default_config_path = (
        f'{default_config_path}/ballistic_default_input.yaml'
    )

    # Create an ArgumentParser instance
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument(
        'config', nargs='?', default=default_config_path,
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
