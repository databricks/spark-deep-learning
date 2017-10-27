#!/usr/bin/env bash
##%%----
## Generated automatically
##%%----

set -eux

_bsd_="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

function quit_with { >&2 echo "ERROR: $@"; exit 1; }
function check_vars {
         for _varname in ${@}; do
                  local _var="$(eval "echo \$${_varname}")"
                  [[ -n "${_var}" ]] || quit_with "${_varname} not defined"
         done
}

##%%----

export SPARK_HOME="${HOME}/code/spark"
PROJECT_HOME="${_bsd_}"
this_package="sparkdl"

_spark_pypath="${SPARK_HOME}"/python:"$(find "${SPARK_HOME}"/python/lib/ -name 'py4j-*-src.zip' -type f | uniq)"
_spark_pkg_pypath="${HOME}/.ivy2/cache/databricks/tensorframes/jars/tensorframes-0.2.9-s_2.11.jar"

_proj_pypath="${PROJECT_HOME}/python"
_py="$(which python2)"

##%%----
check_vars _py
_local_pypath="$(${_py} -c 'import site; print(site.USER_SITE)')"
export PYTHONPATH="${_local_pypath}:${_proj_pypath}:${_spark_pypath}:${_spark_pkg_pypath}"

#pip install --user -r "${_spark_pkg_prereqs}"

(cd ${_bsd_}/python && sphinx-apidoc -f -o docs ${this_package})

pushd "${_bsd_}/docs"
jekyll serve
popd
