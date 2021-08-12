
from setuptools import find_packages
from setuptools import setup
import setuptools

from distutils.command.build import build as _build
import subprocess


REQUIRED_PACKAGES = [
    'transformers',
    'datasets',
    'tqdm',
    'cloudml-hypertune'
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Vertex AI | Training | PyTorch | Text Classification | Python Package'
)
