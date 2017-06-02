---
layout: global
displayTitle: Deep Learning Pipelines Quick-Start Guide
title: Quick-Start Guide
description: Deep Learning Pipelines SPARKDL_VERSION guide for getting started quickly
---

This quick-start guide shows how to get started using Deep Learning Pipelines.
After you work through this guide, move on to the [User Guide](user-guide.html)
to learn more about the many queries and algorithms supported by Deep Learning Pipelines.

* Table of contents
{:toc}

# Getting started with Apache Spark and Spark packages

If you are new to using Apache Spark, refer to the
[Apache Spark Documentation](http://spark.apache.org/docs/latest/index.html) and its
[Quick-Start Guide](http://spark.apache.org/docs/latest/quick-start.html) for more information.

If you are new to using [Spark packages](http://spark-packages.org), you can find more information
in the [Spark User Guide on using the interactive shell](http://spark.apache.org/docs/latest/programming-guide.html#using-the-shell).
You just need to make sure your Spark shell session has the package as a dependency.

The following example shows how to run the Spark shell with the Deep Learning Pipelines package.
We use the `--packages` argument to download the Deep Learning Pipelines package and any dependencies automatically.

<div class="codetabs">

<div data-lang="scala"  markdown="1">

{% highlight bash %}
$ ./bin/spark-shell --packages spark-deep-learning
{% endhighlight %}

</div>

<div data-lang="python"  markdown="1">

{% highlight bash %}
$ ./bin/pyspark --packages spark-deep-learning
{% endhighlight %}

</div>

