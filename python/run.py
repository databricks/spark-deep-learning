#!/usr/bin/env python
"""
This script can be used to run all things dev. Environment setup, Style-checks, Testing etc.
"""

from __future__ import print_function

import argh
from argh import arg
import subprocess
import argcomplete
import sys

def print_if(cond, *args):
    if(cond):
        print(*args)


def call_subprocess(process, keyword_args, trail_args):
    """wrapper function to format kwargs into bash arguments, print and then make the call"""
    single_char_options = [k for k, v in keyword_args.items() if v and len(k) == 1]
    multiple_char_options = [k for k, v in keyword_args.items() if v and len(k) > 1]
    opts = ["-{}{}".format(k.replace("_", "-"), keyword_args[k]) for k in single_char_options]
    opts += ["--{}={}".format(k.replace("_", "-"), keyword_args[k]) for k in multiple_char_options]
    print("calling subprocess: {}".format([process, ] + opts + list(trail_args)))
    subprocess.call([process, ] + opts + list(trail_args))


def add_all_args(all_args, split_on="\n"):
    """decorator with arguments to split a string containing all argh (--argument-strings) and
    decorate a function with all those"""
    def decorator(func):
        for a in set(all_args.split(split_on)):
            if len(a) and a != "--help":
                func = arg(a)(func)
        return func
    return decorator


# list of commands that can be run

# this is the list of arguments pylint can take and can be generated using the following command
# pylint -h | egrep -o '\-\-[0-9a-z\-]+' | sort | uniq
__pylint_flags = """
--confidence
--disable
--enable
--errors-only
--evaluation
--extension-pkg-whitelist
--full-documentation
--generate-rcfile
--help
--help-msg
--ignore
--ignore-patterns
--init-hook
--jobs
--list-conf-levels
--list-msgs
--load-plugins
--long-help
--msg-template
--output-format
--persistent
--py3k
--rcfile
--reports
--score
--suggestion-mode
--version
"""

@add_all_args(__pylint_flags)
def pylint(*args, **kwargs):
    """calls pylint with a limited set of keywords. run `pylint --help` for more details."""
    call_subprocess("pylint", keyword_args=kwargs, trail_args=args)


@add_all_args("")
def prospector(*args, **kwargs):
    """calls prospector with a limited set of keywords. run `prospector --help` for more details."""
    call_subprocess("prospector", keyword_args=kwargs, trail_args=args)


@add_all_args("")
def unittest(*args, **kwargs):
    """calls unittest with a limited set of keywords. run `python -m unittest --help` for more
    details."""
    kwargs["m"] = "unittest"
    call_subprocess("python", keyword_args=kwargs, trail_args=args)


@add_all_args("")
def nose(*args, **kwargs):
    """calls nosettest with a limited set of keywords. run `python -m nose --help` for more
    details."""
    kwargs["m"] = "nose"
    call_subprocess("python", keyword_args=kwargs, trail_args=args)


def envsetup(default=False, interactive=False, missing_only=False, completion=False, verbose=False):
    """
    Prints out shell commands that can be used in terminal to setup the environment.

    This tool inspects the current environment, adds default values, and/or interactively asks
    user for values and prints all or the missing variables that need to be set.

    :param default: if default values should be set in this script or not
    :param interactive: if user should be prompted for values or not
    :param missing_only: if only missing variable should be printed
    :param verbose: if user should be guided or not
    """
    import os
    default_env = {'PYSPARK_PYTHON': 'python', 'SPARK_VERSION': '2.3.0',
                   'SPARK_HOME': '$PATH/spark-2.3.0-bin-hadoop2.7/', 'SCALA_VERSION': '2.11.8'}
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
                new_value = input("{}=".format(k))
                if new_value:
                    env[k] = new_value

    env = {k: v for k, v in env.items() if k in missing_vars or not missing_only}
    env_str = "\n".join("export {}={}".format(k, v) for k, v in env.items())
    if completion:
        env_str += argcomplete.shellcode(sys.argv[0])
    print(env_str)

parser = argh.ArghParser()
args = parser.add_commands([pylint, prospector, unittest, nose, envsetup])

if __name__ == '__main__':
    parser.dispatch()