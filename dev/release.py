#!/usr/bin/env python
import click
from six.moves import input
from subprocess import check_call, check_output

WORKING_BRANCH = "PREP_RELEASE_%s"
DATABRICKS_REMOTE = "git@github.com:databricks/spark-deep-learning.git"

def prominentPrint(x):
    x = str(x)
    n = len(x)
    lines = ["", "=" * n, x, "=" * n, ""]
    print("\n".join(lines))

def askYesNo(prompt):
    response = None
    while response not in ("y", "yes", "n", "no"):
        response = input(prompt).lower()
    return response in ("y", "yes")


@click.command()
@click.argument("release_version", type=str)
@click.argument("next_version", type=str)
@click.option("--local", is_flag=True)
@click.option("--skip-tests", is_flag=True)
def main(release_version, next_version, local, skip_tests):
    if not next_version.endswith("SNAPSHOT"):
        next_version += "-SNAPSHOT"

    if not askYesNo("Publishing version: %s\n"
                    "Next version will be: %s\n"
                    "Continue? (y/n)" % (release_version, next_version)):
        return

    current_branch = check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
    if current_branch != "master":
        if not askYesNo("You're not on the master branch do you want to continue? (y/n)"):
            return

    uncommitted_changes = check_output(["git", "diff", "--stat"])
    if uncommitted_changes != "":
        print(uncommitted_changes)
        print("There seem to be uncommitted changes on your current branch. Please commit or "
              "stash them and try again.")
        return

    working_branch = WORKING_BRANCH % release_version
    if working_branch in check_output(["git", "branch"]):
        prominentPrint(
            "Working branch %s already exists, please delete it and try again." % working_branch)
        return

    prominentPrint("Creating working branch for this release.")
    check_call(["git", "checkout", "-b", working_branch])

    prominentPrint("Creating release tag and updating snapshot version.")
    update_version = "release release-version %s next-version %s" % (release_version, next_version)
    check_call(["./build/sbt", update_version])

    prominentPrint("Building and testing with sbt.")
    check_call(["git", "checkout", "v%s" % release_version])

    if not skip_tests:
        check_call(["./build/sbt", "clean", "test"])

        prominentPrint("Running python tests.")
        check_call(["python/run-tests.sh"])

    if local:
        check_call(["./build/sbt", "publishLocal"])
    else:
        raise NotImplementedError("TODO")

    prominentPrint("Updating local branch: %s" % current_branch)
    check_call(["git", "checkout", current_branch])
    check_call(["git", "merge", "--ff", working_branch])
    check_call(["git", "branch", "-d", working_branch])

    prominentPrint("Local branch updated")
    if askYesNo("Would you like to push local branch to databricks remote? (y/n)"):
        check_call(["git", "push", DATABRICKS_REMOTE, current_branch])


if __name__ == "__main__":
    main()
