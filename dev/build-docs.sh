#!/bin/bash

# build the base image
docker build -t databricks/sparkdl --build-arg PYTHON_VERSION=2.7 .

# build the docs image
docker build -t databricks/sparkdl-docs docs/

# create the assembly jar on host because we have ivy/maven cache
build/sbt assembly

# build the API docs
docker run --rm -v "$(pwd):/mnt/sparkdl" databricks/sparkdl-docs \
    bash -i -c dev/build-docs-in-docker.sh
