#!/usr/bin/env python
"""
This script can be used to run all things dev. Environment setup, Style-checks, Testing etc.
"""

from __future__ import print_function

import os
import subprocess
import sys

import argcomplete
import argh

import pylint
import prospector
import yapf


DIR = os.path.dirname(os.path.realpath(__file__))   # path of directory this file resides in
HOME = os.getenv('HOME', '')    # HOME environment variable


def print_if(cond, *args):
    if (cond):
        print(*args)


def call_subprocess(process, keyword_args, trail_args):
    """wrapper function to format kwargs into bash arguments, print and then make the call"""
    single_char_options = [k for k, v in keyword_args.items() if v and len(k) == 1]
    multiple_char_options = [k for k, v in keyword_args.items() if v and len(k) > 1]
    opts = ["-{}{}".format(k.replace("_", "-"), keyword_args[k]) for k in single_char_options]
    opts += ["--{}={}".format(k.replace("_", "-"), keyword_args[k]) for k in multiple_char_options]
    print("calling subprocess: {}".format([process, ] + opts + list(trail_args)))
    return subprocess.call([process, ] + opts + list(trail_args))


@argh.arg("args", help="""list of files,packages or modules. if nothing is specified,
default value is sparkdl""")
def pylint(rcfile="./python/.pylint/accepted.rc", reports="y", *args):
    """
    Wraps `pylint` and provides defaults. Run `prospector --help` for more details. Trailing
    arguments are a list of files, packages or modules. if nothing is specified, default value is
    ./python/sparkdl
    """
    if not args:
        args = ("./python/sparkdl", )
    kwargs = {k: v for k, v in locals().items() if k != "args" and v}
    kwargs["m"] = "pylint"
    return call_subprocess("python", keyword_args=kwargs, trail_args=args)


def prospector(without_tool="pylint", *args):
    """
    Wraps `prospector` and provides defaults. Run `prospector --help` for more details. Trailing
    arguments are a list of files, packages or modules. if nothing is specified, default value is
    ./python/sparkdl
    """
    if not args:
        args = ("./python/sparkdl", )
    kwargs = {k: v for k, v in locals().items() if k != "args" and v}
    kwargs["m"] = "prospector"
    return call_subprocess("python", keyword_args=kwargs, trail_args=args)


def yapf(style="{based_on_style=pep8, COLUMN_LIMIT=100}", in_place=False, recursive=False, *args):
    """Wraps `yapf` and provides some defaults. Run `yapf --help` for more details."""
    if in_place:
        args = ("-i",) + args
    if recursive:
        args = ("-r",) + args
    return call_subprocess("python", keyword_args={"m": "yapf", "style": style}, trail_args=args)


def envsetup(default=False, interactive=False, missing_only=False, completion=False, verbose=False):
    """
    Prints out shell commands that can be used in terminal to setup the environment.

    This tool inspects the current environment, and/or adds default values, and/or interactively
    asks user for values and prints all or the missing variables that need to be set. It can also
    provide completion for this script. You can source the setup as follows:

    ```
    python/run.py envsetup -d -c > ./python/.setup.sh && source ./python/.setup.sh
    ```

    :param default: if default values should be set in this script or not
    :param interactive: if user should be prompted for values or not
    :param missing_only: if only missing variable should be printed
    :param completion: if auto complete code should be printed
    :param verbose: if user should be guided or not
    """
    env_str = "#!/bin/bash\n"

    default_env = {'PYSPARK_PYTHON': 'python', 'SPARK_VERSION': '2.3.0',
                   'SPARK_HOME': os.path.join(HOME, 'bin/spark-2.3.0-bin-hadoop2.7/'),
                   'SCALA_VERSION': '2.11.8'}
    env = {k: os.environ.get(k, None) for k in default_env.keys()}
    given_vars = [k for k, v in env.items() if v]
    missing_vars = [k for k, v in env.items() if not v]

    print_if(given_vars and verbose, 'given environment variables: {}'.format(given_vars))
    if missing_vars:
        print_if(verbose, 'missing environment variables: {}'.format(missing_vars))

        if default:
            print_if(verbose, 'using default values')
            for k in missing_vars:
                env[k] = default_env[k]

        if interactive:
            print_if(verbose, 'enter values for the following')
            print_if(default and verbose, 'if left blank, default values will be used')
            for k in missing_vars:
                try:
                    new_value = input("{}=".format(k))
                    if new_value:
                        env[k] = new_value
                except SyntaxError:
                    pass

    env = {k: v for k, v in env.items() if k in missing_vars or not missing_only}
    env_str += "\n".join("export {}={}".format(k, v) for k, v in env.items())

    if completion:
        env_str += argcomplete.shellcode(sys.argv[0])

    env_str += """
    # The current directory of the script.
    export DIR={}
    """.format(DIR)

    env_str += """
    LIBS=""
    for lib in "$SPARK_HOME/python/lib"/*zip ; do
      LIBS=$LIBS:$lib
    done
    
    a=( ${SCALA_VERSION//./ } )
    scala_version_major_minor="${a[0]}.${a[1]}"
    assembly_path="$DIR/../target/scala-$scala_version_major_minor"
    JAR_PATH=""
    for assembly in $assembly_path/spark-deep-learning-assembly*.jar ; do
      JAR_PATH=$assembly
    done
    
    # python dir ($DIR) should be before assembly so dev changes can be picked up.
    export PYTHONPATH=$PYTHONPATH:$DIR
    export PYTHONPATH=$PYTHONPATH:$assembly   # same $assembly used for the JAR_PATH above
    export PYTHONPATH=$PYTHONPATH:$SPARK_HOME/python:$LIBS:.
    """
    print(env_str)
    return 0


parser = argh.ArghParser()
parser.add_commands([pylint, prospector, yapf, envsetup])

if __name__ == '__main__':
    dispatch_code = parser.dispatch(output_file=None)
    # argh.dispatch formats the returns of functions as strings
    exit(int(dispatch_code))
