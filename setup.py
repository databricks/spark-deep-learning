#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" distribute- and pip-enabled setup.py """

import setuptools
import sys

import sparkdl

exclude_packages = ('tests', 'tests.*')

if '--with-tests' in sys.argv:
    index = sys.argv.index('--with-tests')
    sys.argv.pop(index)
    exclude_packages = ()

setuptools.setup(
    name='sparkdl',
    version=sparkdl.__version__,
    packages=setuptools.find_packages(exclude=exclude_packages),
    url="https://github.com/databricks/spark-deep-learning",
    author="Weichen Xu",
    author_email="weichen.xu@databricks.com",
    description="Deep Learning Pipelines for Apache Spark",
    long_description="",
    classifiers=[
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Environment :: Console",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    platforms=["Linux"],
    license="BSD",
    keywords="spark deep learning horovod model distributed training",
    install_requires=[],
    extras_require={},
    tests_require=["nose", "pytest"],
    zip_safe=False,
)
