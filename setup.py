# -*- coding: utf-8 -*-
"""
setup.py — packaging for the spot package.

spot is an integrated Stokes spectral synthesis and inversion library
(forward synthesis + response functions + node-based LM / CMA-ES
inversion) with a bundled neural-operator subpackage ``spot.net`` for
network-based Stokes inversion.  Everything runs with pure Python /
PyTorch (Numpy) on CPU or CUDA.

Author  : Guoyin Chen
Email   : gychen@smail.nju.edu.cn
"""

import os

from setuptools import find_packages, setup


def collect_data_files(root):
    """Return [(package_dir, [files...])] for every data file under ``root``."""
    entries = []
    for dirpath, _dirnames, filenames in os.walk(root):
        files = [f for f in filenames
                 if f.endswith((".csv", ".txt", ".json", ".pt", ".mod"))]
        if not files:
            continue
        pkg = dirpath.replace(os.sep, ".")
        entries.append((pkg, [os.path.join(dirpath, f) for f in files]))
    return entries


# get the long description from the README (bilingual, UTF-8)
with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="spot",
    version="1.0.0",
    description=(
        "Integrated Stokes spectral synthesis and inversion library: "
        "batched GPU/CPU synthesis of Stokes IQUV, response functions, "
        "node-based Levenberg-Marquardt and CMA-ES inversion, plus the "
        "spot.net neural-operator subpackage for network-based Stokes "
        "inversion with pretrained models."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Guoyin Chen",
    author_email="gychen@smail.nju.edu.cn",
    url="https://github.com/gychen-NJU/SPOT",
    license="MIT",
    packages=find_packages(include=("spot", "spot.*")),
    package_data={
        "spot": [
            "data/*.csv",
            "data/models/*.csv",
            "data/opacity_tables/*.csv",
            "data/opacity_tables/*.txt",
            "data/opacity_tables/OPtabs/*.csv",
        ],
        "spot.net": [
            "models/hinode_sp/*.pt",
            "models/hinode_sp/*.json",
        ],
    },
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
        "torch>=2.0",
        "matplotlib>=3.5",
    ],
    extras_require={
        "test": ["pytest>=7.0"],
        "plot": ["matplotlib>=3.5"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    zip_safe=False,
)
