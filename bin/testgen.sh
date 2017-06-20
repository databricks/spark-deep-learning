#!/bin/bash

#######################################################
scala_major_ver=2.11
python_package_name="sparkdl"
#######################################################

# The script rest in under the <project_root>/bin directory
_bsd_="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."

function quit_with { >&2 echo "ERROR: $@"; exit 1; }

[[ -n "${SPARK_HOME}" ]] || \
    quit_with "must provide Spark home"

host_spark_home="${SPARK_HOME}"

# Generate required classpaths
sbt_path_root="${_bsd_}/.sbt.paths"
(cd "${_bsd_}"
 rm -rf "${sbt_path_root}" && mkdir -p $_
 if [[ -x ./sbt ]]; then
     ./sbt assembly genClasspath
 elif [[ -x ./build/sbt ]]; then
     ./build/sbt assembly genClasspath
 else
     quit_with "cannot find workable sbt"
 fi
)

[[ -f "${sbt_path_root}/SBT_SPARK_PACKAGE_CLASSPATH" ]] || \
    quit_with "please run: './sbt genClasspath'"

[[ -f "${sbt_path_root}/SBT_RUNTIME_CLASSPATH" ]] || \
    quit_with "please run: './sbt genClasspath'"

# Spark packages are cached in local ivy cache
(cd "${sbt_path_root}"
 rm -f SPARK_PACKAGE_PYREQ && touch $_
 for spkg in $(cat SBT_SPARK_PACKAGE_CLASSPATH | tr ':' '\n'); do
     echo $spkg
     printf "\n# BEGIN: $(basename $spkg)\n" >> SPARK_PACKAGE_PYREQ
     unzip -p $spkg "$(jar -tf "${spkg}" | grep -Ei 'requirements.txt')" >> SPARK_PACKAGE_PYREQ
     printf "\n# END: $(basename $spkg)\n" >> SPARK_PACKAGE_PYREQ
 done

 pip2 install --user -r SPARK_PACKAGE_PYREQ

 perl -pe "s@${HOME}@~@g" SBT_SPARK_PACKAGE_CLASSPATH > SPARK_PACKAGE_PYTHONPATH
 perl -pe "s@${HOME}@~@g" SBT_RUNTIME_CLASSPATH > SPARK_RUNTIME_CLASSPATH
)

_test_script="${_bsd_}/.py.test"
rm -f "${_test_script}" && touch $_ && chmod +x $_

_test_logic_script="${_bsd_}/python/.py.test.logic"
rm -f "${_test_logic_script}" && touch $_ && chmod +x $_

pip2 install --user -r ${_bsd_}/python/requirements.txt

# First define the variables
cat << _PY_TEST_EOF_ | tee -a "${_test_script}"
#!/usr/bin/env bash

#######################################################
scala_major_ver=${scala_major_ver}
sbt_path_root=${sbt_path_root}
py_test_logic="$(basename "${_test_logic_script}")"
export SPARK_HOME=${host_spark_home}
#######################################################

_PY_TEST_EOF_

# Then define the main script
cat << '_PY_TEST_EOF_' | tee -a "${_test_script}"
_bsd_="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

assembly_path="${_bsd_}/target/scala-${scala_major_ver}"
assembly_jar="$(find ${assembly_path} -name "*-assembly*-SNAPSHOT-*.jar" -type f | uniq | head)"

# Translate the ~ to absolute paths
function tr_classpath { perl -pe "s@~@$HOME@g" "${sbt_path_root}/$@"; }
function mk_classpath { local IFS=':'; echo "$*"; }

export PYSPARK_PYTHON="$(which python)"
export PYSPARK_DRIVER_PYTHON=$PYSPARK_PYTHON

# Setup python paths
_local_pypath="${_bsd_}/python:$(python -c 'import site; print(site.USER_SITE)')"
_spark_pkg_path="$(tr_classpath SPARK_PACKAGE_PYTHONPATH)"
_spark_pypath="${SPARK_HOME}"/python:"$(find "${SPARK_HOME}"/python/lib/ -name 'py4j-*-src.zip' -type f | uniq)"

export PYTHONPATH="${_local_pypath}:${_spark_pkg_path}:${_spark_pypath}"
echo $PYTHONPATH

# Also need some of the SBT runtime jars for py4j to work
_sbt_runtime_jars="$(tr_classpath SPARK_RUNTIME_CLASSPATH | tr ':' '\n' | grep -Ei '.jar$')"
_submit_py_files="$(echo "${assembly_jar}:${_spark_pkg_path}" | tr ':' ',')"
_submit_jars="$(echo "${assembly_jar}:${_spark_pkg_path}:$(mk_classpath ${_sbt_runtime_jars})" | tr ':' ',')"

# Provide required python and jar files for Spark testing cluster
export PYSPARK_SUBMIT_ARGS="--verbose --driver-memory 4g --executor-memory 4g --py-files ${_submit_py_files} --jars ${_submit_jars} pyspark-shell"

pushd ${_bsd_}/python
exec "./${py_test_logic}" $@
popd
_PY_TEST_EOF_


cat << _PY_TEST_LOGIC_HEADER_EOF_ | tee -a "${_test_logic_script}"
#!/usr/bin/env bash

#######################################################
this_py_pkg=${python_package_name}
#######################################################
_PY_TEST_LOGIC_HEADER_EOF_

cat << '_PY_TEST_LOGIC_EOF_' | tee -a "${_test_logic_script}"
#!/usr/bin/env bash
# Return on any failure
set -e

# TODO: make sure travis has the right version of nose
if [[ $# -gt 0 ]]; then
    noseOptionsArr=()
    for tf in ${@}; do noseOptionsArr+=("tests/$tf"); done
else
    # add all python files in the test dir recursively
    noseOptionsArr=($(find tests -type f | grep -E "\.py$" | grep -v "__init__.py"))
fi

# Limit TensorFlow error message
# https://github.com/tensorflow/tensorflow/issues/1258
export TF_CPP_MIN_LOG_LEVEL=3

for noseOptions in ${noseOptionsArr[@]}; do
  echo "============= Running the tests in: $noseOptions ============="

  # The grep below is a horrible hack for spark 1.x:
  # => we manually remove some log lines to stay below the 4MB log limit on Travis.
  $PYSPARK_DRIVER_PYTHON \
      -m "nose" \
      --with-coverage --cover-package="${this_py_pkg}" \
      --nologcapture \
      -v --exe "$noseOptions" \
      2>&1 | grep -vE "INFO (ParquetOutputFormat|SparkContext|ContextCleaner|ShuffleBlockFetcherIterator|MapOutputTrackerMaster|TaskSetManager|Executor|MemoryStore|CacheManager|BlockManager|DAGScheduler|PythonRDD|TaskSchedulerImpl|ZippedPartitionsRDD2)";

  # Exit immediately if the tests fail.
  # Since we pipe to remove the output, we need to use some horrible BASH features:
  # http://stackoverflow.com/questions/1221833/bash-pipe-output-and-capture-exit-status
  test ${PIPESTATUS[0]} -eq 0 || exit 1;
done


# Run doc tests

#$PYSPARK_PYTHON -u ourpythonfilewheneverwehaveone.py "$@"
_PY_TEST_LOGIC_EOF_


echo "##############################################"
echo "# TEST script generated: ${_test_script}"
echo "# TEST Python logic script generated: ${_test_logic_script}"
exec "${_test_script}" $@
