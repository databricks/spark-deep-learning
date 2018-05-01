#!/usr/bin/env python
"""
This script can be used to run all things dev. Environment setup, Style-checks, Testing etc.
"""

from __future__ import print_function

import os
import subprocess
import sys
import six

import argcomplete
import argh

import pylint as _pylint
import prospector as _prospector
import yapf as _yapf


DIR = os.path.dirname(os.path.realpath(__file__))   # path of directory this file resides in
HOME = os.getenv('HOME', '')    # HOME environment variable


def _list_files_with_extension(dir_name, ext):
    return [f for f in os.listdir(dir_name) if f.endswith(ext)]


def _get_configured_env(_required_env):
    spark_home = _required_env["SPARK_HOME"]
    scala_version = _required_env["SCALA_VERSION"]

    if not spark_home:
        raise ValueError("set SPARK_HOME environment variable to configure environment")
    if not scala_version:
        raise ValueError("set SCALA_VERSION environment variable to configure environment")

    configured_env = {"DIR": DIR}

    spark_lib_path = os.path.join(spark_home, "python/lib")
    configured_env["LIBS"] = ":".join(_list_files_with_extension(spark_lib_path, ".zip"))

    scala_version_major_minor = ".".join(scala_version.split(".")[0:2])
    assembly_path = os.path.join(DIR, "../target/scala-" + scala_version_major_minor)
    jar_path = ":".join(_list_files_with_extension(assembly_path, ".jar"))
    configured_env["JAR_PATH"] = jar_path

    python_paths = [
        os.getenv("PYTHON_PATH", ""),
        DIR,
        _list_files_with_extension(assembly_path, ".jar")[-1],
        os.path.join(spark_home, "python"),
        configured_env["LIBS"]
    ]
    configured_env["PYTHON_PATH"] = ":".join(python_paths)

    return configured_env


def configure_env():
    current_env = dict(os.environ)
    configured_env = _get_configured_env(current_env)
    for k, v in configured_env.items():
        if six.PY2:
            os.environ[k] = v
        else:
            os.environ.putenv(k, v)


def call_subprocess(process, keyword_args, trail_args):
    """wrapper function to format kwargs into bash arguments, print and then make the call"""
    single_char_options = [k for k, v in keyword_args.items() if v and len(k) == 1]
    multiple_char_options = [k for k, v in keyword_args.items() if v and len(k) > 1]
    opts = ["-{}{}".format(k.replace("_", "-"), keyword_args[k]) for k in single_char_options]
    opts += ["--{}={}".format(k.replace("_", "-"), keyword_args[k]) for k in multiple_char_options]
    print("calling subprocess: {}".format([process, ] + opts + list(trail_args)))
    configure_env()
    return subprocess.call([process, ] + opts + list(trail_args))


def pylint(rcfile="./python/.pylint/accepted.rc", reports="y", persistent="n", *args):
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


def pylint_suggested(*args):
    """Wrapper for pylint that is used to generate suggestions"""
    return pylint(rcfile="./python/.pylint/suggested.rc", reports="n", persistent="n", *args)


def prospector(without_tool="pylint", output_format="pylint", *args):
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


def tests(*args):
    """Wrapper for run-tests.sh"""
    return call_subprocess("./python/run-tests.sh", keyword_args={}, trail_args=args)


def _safe_input_prompt(k, default):
    try:
        current = input("{}=".format(k))
    except SyntaxError:
        # this error is thrown in python 2.7
        # SyntaxError: unexpected EOF while parsing
        current = ""
    return current if current else default


def print_if(cond, *args):
    if cond:
        print(*args)


def _env2shellcode(env):
    # Dict[str, str] -> str
    return "\n".join("export {}={}".format(k, v) for k, v in env.items())


def envsetup(default=False, interactive=False, missing_only=False, override=False, configure=False,
             auto_completion=False, verbose=False):
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
    :param override: if current environment variables should be overridden
    :param configure: if configuration should be printed or not
    :param auto_completion: if auto complete code should be printed
    :param verbose: if user should be guided or not
    """
    default_env = {'PYSPARK_PYTHON': 'python',
                   'SPARK_VERSION': '2.3.0',
                   'SPARK_HOME': os.path.join(HOME, 'bin/spark-2.3.0-bin-hadoop2.7/'),
                   'SCALA_VERSION': '2.11.8'}

    if override and not (interactive or default):
        raise ValueError("override mode requires to use default or interactive mode")

    # identify which required variables are given and which are missing
    given_env = {}
    missing_env = {}
    for k in default_env:
        v = os.getenv(k, "")
        if v and not override:
            given_env[k] = v
        else:
            missing_env[k] = ""
    print_if(given_env and verbose, 'given environment variables: {}'.format(given_env))

    # set the missing variables interactively or from defaults
    if missing_env:
        print_if(verbose, 'missing environment variables: {}'.format(missing_env))

        if not (default or interactive):
            raise ValueError(
                """Use default or interactive mode to set required environment variables: 
                {}""".format(",".join(missing_env)))

        if default:
            print_if(verbose, 'using default values')
            missing_env = {k: default_env[k] for k in missing_env}

        if interactive:
            print_if(verbose, 'enter values for the following')
            print_if(default and verbose, 'if left blank, default values will be used')
            missing_env = {k: _safe_input_prompt(k, v) for k, v in missing_env.items()}

    # print according to options
    output_env = {}
    output_env.update(missing_env)

    if not missing_only:
        output_env.update(given_env)

    if configure:
        required_env = dict(given_env)
        required_env.update(missing_env)
        configured_env = _get_configured_env(required_env)
        output_env.update(configured_env)

    env_str = "#!/bin/bash\n"
    env_str += _env2shellcode(output_env)

    if auto_completion:
        env_str += argcomplete.shellcode(sys.argv[0])

    print(env_str)
    return 0


parser = argh.ArghParser()
parser.add_commands([pylint, prospector, yapf, envsetup, tests, pylint_suggested])

if __name__ == '__main__':
    dispatch_result = parser.dispatch(output_file=None)
    # argh.dispatch formats the returns of functions as strings
    try:
        return_code = int(dispatch_result)
    except ValueError:
        print(dispatch_result)
        return_code = 0
    exit(return_code)

