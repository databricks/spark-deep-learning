#!/usr/bin/env bash

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Return on any failure
set -e

# if (got > 1 argument OR ( got 1 argument AND that argument does not exist)) then
# print usage and exit.
if [[ $# -gt 1 || ($# = 1 && ! -e $1) ]]; then
  echo "run_tests.sh [target]"
  echo ""
  echo "Run python tests for this package."
  echo "  target -- either a test file or directory [default tests]"
  if [[ ($# = 1 && ! -e $1) ]]; then
    echo
    echo "ERROR: Could not find $1"
  fi
  exit 1
fi

# assumes run from python/ directory
if [ -z "$SPARK_HOME" ]; then
    echo 'You need to set $SPARK_HOME to run these tests.' >&2
    exit 1
fi

# Honor the choice of python driver
if [ -z "$PYSPARK_PYTHON" ]; then
    PYSPARK_PYTHON=`which python`
fi
# Override the python driver version as well to make sure we are in sync in the tests.
export PYSPARK_DRIVER_PYTHON=$PYSPARK_PYTHON
python_major=$($PYSPARK_PYTHON -c 'import sys; print(".".join(map(str, sys.version_info[:1])))')

echo $pyver

LIBS=""
for lib in "$SPARK_HOME/python/lib"/*zip ; do
  LIBS=$LIBS:$lib
done

# The current directory of the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

a=( ${SCALA_VERSION//./ } )
scala_version_major_minor="${a[0]}.${a[1]}"
echo "List of assembly jars found, the last one will be used:"
assembly_path="$DIR/../target/scala-$scala_version_major_minor"
echo `ls $assembly_path/spark-deep-learning-assembly*.jar`
JAR_PATH=""
for assembly in $assembly_path/spark-deep-learning-assembly*.jar ; do
  JAR_PATH=$assembly
done

# python dir ($DIR) should be before assembly so dev changes can be picked up.
export PYTHONPATH=$PYTHONPATH:$DIR
export PYTHONPATH=$PYTHONPATH:$assembly   # same $assembly used for the JAR_PATH above
export PYTHONPATH=$PYTHONPATH:$SPARK_HOME/python:$LIBS:.

# This will be used when starting pyspark.
export PYSPARK_SUBMIT_ARGS="--driver-memory 4g --executor-memory 4g --jars $JAR_PATH pyspark-shell"


# Run test suites

# TODO: make sure travis has the right version of nose
if [ -f "$1" ]; then
  noseOptionsArr="$1"
else
  if [ -d "$1" ]; then
    targetDir=$1
  else
    targetDir=$DIR/tests
  fi
  # add all python files in the test dir recursively
  echo "============= Searching for tests in: $targetDir ============="
  noseOptionsArr="$(find "$targetDir" -type f | grep "\.py" | grep -v "\.pyc" | grep -v "\.py~" | grep -v "__init__.py")"
fi

# Limit TensorFlow error message
# https://github.com/tensorflow/tensorflow/issues/1258
export TF_CPP_MIN_LOG_LEVEL=3

for noseOptions in $noseOptionsArr
do
  echo "============= Running the tests in: $noseOptions ============="
  # The grep below is a horrible hack for spark 1.x: we manually remove some log lines to stay below the 4MB log limit on Travis.
  $PYSPARK_DRIVER_PYTHON \
      -m "nose" \
      --with-coverage --cover-package=sparkdl \
      --nologcapture \
      -v --exe "$noseOptions" \
      2>&1 | grep -vE "INFO (ParquetOutputFormat|SparkContext|ContextCleaner|ShuffleBlockFetcherIterator|MapOutputTrackerMaster|TaskSetManager|Executor|MemoryStore|CacheManager|BlockManager|DAGScheduler|PythonRDD|TaskSchedulerImpl|ZippedPartitionsRDD2)";

  # Exit immediately if the tests fail.
  # Since we pipe to remove the output, we need to use some horrible BASH features:
  # http://stackoverflow.com/questions/1221833/bash-pipe-output-and-capture-exit-status
  test ${PIPESTATUS[0]} -eq 0 || exit 1;
done


# Run doc tests

#$PYSPARK_PYTHON -u ./sparkdl/ourpythonfilewheneverwehaveone.py "$@"
