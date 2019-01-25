Welcome to the Deep Learning Pipelines Spark Package documentation!

This readme will walk you through navigating and building the Deep Learning Pipelines documentation, which is
included here with the source code.

Read on to learn more about viewing documentation in plain text (i.e., markdown) or building the
documentation yourself. Why build it yourself? So that you have the docs that correspond to
whichever version of Deep Learning Pipelines you currently have checked out of revision control.

We include the Deep Learning Pipelines documentation as part of the source (as opposed to using a hosted wiki, such as
the github wiki, as the definitive documentation) to enable the documentation to evolve along with
the source code and be captured by revision control (currently git). This way the code automatically
includes the version of the documentation that is relevant regardless of which version or release
you have checked out or downloaded.

In this directory you will find textfiles formatted using Markdown, with an ".md" suffix. You can
read those text files directly if you want. Start with index.md.

## Generating the Documentation HTML
To generate the documentation HTML, you can run the script from the base directory:

    $ dev/build-docs.sh

It compiles the site with Jekyll will create a directory called `_site` containing index.html as well as the rest of the compiled files.

You can modify the default Jekyll build in the docs docker container as follows:

    # Skip generating API docs (which takes a while)
    $ SKIP_API=1 ./jekyll build
    # Serve content locally on port 4000
    $ ./jekyll serve --watch
    # Build the site with extra features used on the live page
    $ PRODUCTION=1 ./jekyll build

## Pygments

We also use pygments (http://pygments.org) for syntax highlighting in documentation markdown pages.
To mark a block of code in your markdown to be syntax highlighted by jekyll during the compile
phase, use the following sytax:

    {% highlight scala %}
    // Your scala code goes here, you can replace scala with many other
    // supported languages too.
    {% endhighlight %}

## Sphinx

We use Sphinx to generate Python API docs.

## API Docs (Scaladoc, Sphinx)

You can build just the scaladoc by running `build/sbt unidoc` from the SPARKDL_PROJECT_ROOT directory.

Similarly, you can build just the Python docs by running `make html` from the
SPARKDL_PROJECT_ROOT/python/docs directory. Documentation is only generated for classes that are listed as
public in `__init__.py`.

When you run `jekyll` in the `docs` directory, it will also copy over the scaladoc for the various
subprojects into the `docs` directory (and then also into the `_site` directory). We use a
jekyll plugin to run `build/sbt unidoc` before building the site so if you haven't run it (recently) it
may take some time as it generates all of the scaladoc.  The jekyll plugin also generates the
Python docs [Sphinx](http://sphinx-doc.org/).

NOTE: To skip the step of building and copying over the Scala, Python API docs, run `SKIP_API=1
jekyll build`. To skip building Scala API docs, run `SKIP_SCALADOC=1 jekyll build`; to skip building Python API docs, run `SKIP_PYTHONDOC=1 jekyll build`.
