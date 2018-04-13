#!/usr/bin/env python

import argparse, argh

def print_if(cond, *args):
    if(cond):
        print(*args)

def pylint(*args, **kwargs):
    print(args, kwargs)


def prospector(*args, **kwargs):
    print(args, kwargs)


def unittest(*args, **kwargs):
    print(args, kwargs)


def nosettest(*args, **kwargs):
    print(args, kwargs)


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

    env_str = ["export {}={}".format(k, v) for k, v in env.items()
               if k in missing_vars or not missing_only]
    print("\n".join(env_str))
    return 0

parser = argh.ArghParser()
args = parser.add_commands([pylint, prospector, unittest, nosettest, envsetup])

if __name__ == '__main__':
    parser.dispatch()