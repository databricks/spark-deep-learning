#!/bin/bash

#######################################################
# Create the list of necessary environment variables
scala_major_ver=2.11
#######################################################

# The current directory of the script.
_bsd_="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."

function quit_with { >&2 echo "$(tput setaf 1)ERROR: $@$(tput sgr0)"; exit 1; }

[[ -n "${SPARK_HOME}" ]] || \
    quit_with "must provide Spark home"

host_spark_home="${SPARK_HOME}"

# Generate required classpaths
[[ -x "${_bsd_}/sbt" ]] || \
    quit_with "cannot locate runnable project sbt executable"

sbt_path_root="${_bsd_}/.sbt.paths"
function tr_classpath { perl -pe "s@~@$HOME@g" "${sbt_path_root}/$@"; }
function mk_classpath { local IFS=':'; echo "$*"; }

# Spark packages are cached in local ivy cache
(cd "${_bsd_}"
 #./sbt genClasspath assembly
 ./sbt genClasspath
 cd "${sbt_path_root}"
 rm -f SPARK_PACKAGE_PYREQ && touch $_
 for spkg in $(cat SBT_SPARK_PACKAGE_CLASSPATH | tr ':' '\n'); do 
     echo "[py-deps]: ${spkg}"
     printf "\n# BEGIN: $(basename $spkg)\n" >> SPARK_PACKAGE_PYREQ
     unzip -p "${spkg}" "$(jar -tf "${spkg}" | grep -Ei 'requirements.txt')" >> SPARK_PACKAGE_PYREQ
     printf "\n# END: $(basename $spkg)\n" >> SPARK_PACKAGE_PYREQ    
 done
 perl -pe "s@${HOME}@~@g" SBT_SPARK_PACKAGE_CLASSPATH > SPARK_PACKAGE_CLASSPATH
 perl -pe "s@${HOME}@~@g" SBT_RUNTIME_CLASSPATH > SPARK_RUNTIME_EXTRA_CLASSPATH
 rm -f SBT_*_CLASSPATH
)

# Find the local assembly jar, if any
_sbt_target_path="${_bsd_}/target/scala-${scala_major_ver}"
assembly_jar="$(find ${_sbt_target_path} -name "*-assembly*.jar" -type f | uniq)"

# Setup python paths
_proj_pypath="${_bsd_}"/python
_spark_pypath="${SPARK_HOME}"/python:"$(find "${SPARK_HOME}"/python/lib/ -name 'py4j-*-src.zip' -type f | uniq)"
_spark_pkg_path="$(tr_classpath SPARK_PACKAGE_CLASSPATH)"
_spark_pkg_pyreq="${sbt_path_root}/SPARK_PACKAGE_PYREQ"

# Also need some of the SBT runtime jars for py4j to work
_sbt_runtime_jars="$(tr_classpath SPARK_RUNTIME_EXTRA_CLASSPATH | tr ':' '\n' | grep -Ei '.jar$')"
_submit_py_files="$(echo "${assembly_jar}:${_spark_pkg_path}" | tr ':' ',')"
_submit_jars="$(echo "${assembly_jar}:${_spark_pkg_path}:$(mk_classpath ${_sbt_runtime_jars})" | tr ':' ',')"

# Provide required python and jar files for Spark testing cluster
# Create individial scripts

#######################################################
# PySpark
#######################################################

SCPT="${_bsd_}/.EXEC_SCRIPT"

function SCPT_BEGIN { 
    rm -f "${SCPT}" && touch $_ && chmod +x $_
    cat << '_SCPT_HEADER_EOF_' | tee -a "${SCPT}"
#!/usr/bin/env bash
################################################
## Generated automatically
################################################

_bsd_="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

function quit_with { >&2 echo "ERROR: $@"; exit 1; }
function check_vars { 
         for _varname in ${@}; do
                  local _var="$(eval "echo \$${_varname}")"
                  [[ -n "${_var}" ]] || quit_with "${_varname} not defined"
         done
}

_SCPT_HEADER_EOF_

cat << _SCPT_HEADER_VAR_EOF_ | tee -a "${SCPT}"
################################################
# Global variables

spark_pkg_path="${_spark_pkg_path}"

################################################
_SCPT_HEADER_VAR_EOF_
}

function SCPT_PYSPARK_BODY {
    cat << _SCPT_LOCAL_VAR_EOF_ | tee -a "${SCPT}"
################################################
# Local variables

_proj_pypath="${_proj_pypath}"
_spark_pypath="${_spark_pypath}"
_spark_pkg_pypath="${_spark_pkg_path}"
_submit_jars="${_submit_jars}"
_submit_py_files="${_submit_py_files}"

################################################
_SCPT_LOCAL_VAR_EOF_

    cat << '_EXEC_SCRIPT_EOF_' | tee -a "${SCPT}"
check_vars _py _ipy _pyspark

_local_pypath="$(${_py} -c 'import site; print(site.USER_SITE)')"

export PYSPARK_PYTHON=${_py}
export PYSPARK_DRIVER_PYTHON=${_ipy}
export PYSPARK_DRIVER_PYTHON_OPTS="-i --simple-prompt --pprint"
# We should only be using the assembly to make sure consistency
# REPL based development could always go and reload pieces
#export PYTHONPATH="${_local_pypath}:${_proj_pypath}:${_spark_pypath}:${_spark_pkg_pypath}"
export PYTHONPATH="${_local_pypath}:${_spark_pypath}:${_spark_pkg_pypath}"

exec "${_pyspark}" \
     --master "local[4]" \
     --conf spark.app.name="[drgscl]::pyspark" \
     --conf spark.eventLog.enabled=false \
     --conf spark.driver.memory=10g \
     --conf spark.executor.memory=10g \
     --py-files "${_submit_py_files}" \
     --jars "${_submit_jars}" \
     $@

_EXEC_SCRIPT_EOF_
}


function SCPT_SPARK_SHELL_BODY {
    cat << '_EXEC_SCRIPT_EOF_' | tee -a "${SCPT}"
check_vars _spark_shell
check_vars _submit_jars

exec "${_spark_shell}" \
     --master "local[4]" \
     --conf spark.app.name="[drgscl]::spark-shell" \
     --conf spark.eventLog.enabled=false \
     --jars "${_submit_jars}" \
     $@

_EXEC_SCRIPT_EOF_
}


function gen_jekyll {
    SCPT_BEGIN
    cat << _SCPT_LOCAL_VAR_EOF_ | tee -a "${SCPT}"
################################################
# Local variables

_proj_pypath="${_proj_pypath}"
_spark_pypath="${_spark_pypath}"
_spark_pkg_pypath="${_spark_pkg_path}"
_spark_pkg_prereqs="${_spark_pkg_pyreq}"
export SPARK_HOME="${host_spark_home}"
_py="$(which python)"
_pip="$(which pip)"

################################################
_SCPT_LOCAL_VAR_EOF_

    cat << '_EXEC_SCRIPT_EOF_' | tee -a "${SCPT}"
check_vars _py _pip
_local_pypath="$(${_py} -c 'import site; print(site.USER_SITE)')"
export PYTHONPATH="${_local_pypath}:${_proj_pypath}:${_spark_pypath}:${_spark_pkg_pypath}"

pip install --user -r "${_spark_pkg_prereqs}"

(cd ${_bsd_}/python && sphinx-apidoc -f -o docs neurospark)

pushd "${_bsd_}/docs"
jekyll $@
popd

_EXEC_SCRIPT_EOF_
    
    mv "${SCPT}" ${_bsd_}/.jekyll
}

function gen_py2_spark_shell {
    SCPT_BEGIN
    cat << _SCPT_VAR_EOF_ | tee -a "${SCPT}"
_py="$(which python2)"
_ipy="$(which ipython2)"
_pyspark="${SPARK_HOME}"/bin/pyspark
_SCPT_VAR_EOF_
    SCPT_PYSPARK_BODY
    mv "${SCPT}" ${_bsd_}/.py2.spark.shell
}

function gen_py3_spark_shell {
    SCPT_BEGIN
    cat << _SCPT_VAR_EOF_ | tee -a "${SCPT}"
_py="$(which python3)"
_ipy="$(which ipython3)"
_pyspark="${SPARK_HOME}"/bin/pyspark
_SCPT_VAR_EOF_
    SCPT_PYSPARK_BODY
    mv "${SCPT}" ${_bsd_}/.py3.spark.shell
}

function gen_spark_shell {
    SCPT_BEGIN
    cat << _SCPT_VAR_EOF_ | tee -a "${SCPT}"
_spark_shell="${SPARK_HOME}"/bin/spark-shell
_submit_jars="$(echo "${assembly_jar}:${_spark_pkg_path}" | tr ':' ',')"
_SCPT_VAR_EOF_
    SCPT_SPARK_SHELL_BODY
    mv "${SCPT}" ${_bsd_}/.spark.shell
}

gen_py2_spark_shell
gen_py3_spark_shell
gen_spark_shell
gen_jekyll
