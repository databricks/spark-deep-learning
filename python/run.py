#!/usr/bin/env python
"""
This script can be used to run all things dev. Environment setup, Style-checks, Testing etc.
"""

from __future__ import print_function

import argh
from argh import arg
import subprocess

def print_if(cond, *args):
    if(cond):
        print(*args)

# pylint -h | egrep -o '\-\-[0-9a-z\-]+'
__pylint_flags = """
--help
--help-msg
--version
--help
--long-help
--rcfile
--init-hook
--errors-only
--py3k
--ignore
--ignore-patterns
--persistent
--load-plugins
--jobs
--extension-pkg-whitelist
--suggestion-mode
--help-msg
--list-msgs
--list-conf-levels
--full-documentation
--generate-rcfile
--confidence
--enable
--disable
--disable
--disable
--disable
--enable
--disable
--enable
--disable
--output-format
--reports
--evaluation
--score
--msg-template
 """

def pylint(*args, **kwargs):
    opts = ["--{}={}".format(k.replace("_", "-"), v) for k, v in kwargs.items() if v]
    subprocess.call(["pylint",] + opts + list(args))


for f in set(__pylint_flags.split("\n")):
    f = f.strip()
    if len(f) and f != "--help":
        pylint = arg(f)(pylint)


def prospector(*args, **kwargs):
    opts = ["--{}={}".format(k.replace("_", "-"), v) for k, v in kwargs.items() if v]
    subprocess.call(["prospector", ] + opts + list(args))


def unittest(*args):
    print(args)


def nosettest(*args):
    print(args)


def envsetup(default=False, interactive=False, missing_only=False, verbose=False):
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
    print(env_str)

parser = argh.ArghParser()
args = parser.add_commands([pylint, prospector, unittest, nosettest, envsetup])

if __name__ == '__main__':
    parser.dispatch()