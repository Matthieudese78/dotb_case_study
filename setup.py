#!/usr/bin/python3
from __future__ import annotations

import os
import sys

from setuptools import find_packages
from setuptools import setup

# Test if execution in the local virtual environment :
if 'VIRTUAL_ENV' in os.environ:
    print('Running in virtual environment')
else:
    print('NOT running in virtual environment')
    print(
        '  Consider running : python3 -m venv path_to_root_directory/myenv'
        '                     source path_to_root_directory/myenv/bin/activate',
    )
    sys.exit()

# %% setup function :
# parsing requirements.txt :
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='your-project-name',
    version='0.1.0',
    packages=find_packages(),
    tests_require=['pytest'],
    install_requires=install_requires,
    extras_require={
        'dev': [
            # Development dependencies
        ],
    },
)
