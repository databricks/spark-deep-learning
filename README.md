Deep Learning Pipelines for Apache Spark
============================================================
[![Build Status][pkg-build-badge]][pkg-build-link] [![Coverage][pkg-cov-badge]][pkg-cov-link]

  [pkg-build-badge]: https://travis-ci.org/databricks/spark-deep-learning.svg?branch=master
  [pkg-build-link]: https://travis-ci.org/databricks/spark-deep-learning
  [pkg-cov-badge]: https://codecov.io/gh/databricks/spark-deep-learning/coverage.svg?branch=master
  [pkg-cov-link]: https://codecov.io/gh/databricks/spark-deep-learning/branch/master

Deep Learning Pipelines provides high-level APIs for scalable deep learning in Python with Apache Spark.

- [Overview](#overview)
- [Spark version compatibility](#spark-version-compatibility)
- [Support](#support)
- [Releases](#releases)
- [License](#license)


## Overview

Deep Learning Pipelines provides high-level APIs for scalable deep learning in Python with Apache Spark.

The library comes from Databricks and leverages Spark for its two strongest facets:

1.  In the spirit of Spark and [Spark MLlib](https://spark.apache.org/mllib/), it provides easy-to-use APIs that enable deep learning in very few lines of code.
2.  It uses Spark's powerful distributed engine to scale out deep learning on massive datasets.

Currently, TensorFlow and TensorFlow-backed Keras workflows are supported, with a focus on:
* large-scale inference / scoring
* transfer learning and hyperparameter tuning on image data

Furthermore, it provides tools for data scientists and machine learning experts to turn deep learning models into SQL functions that can be used by a much wider group of users. It does not perform single-model distributed training - this is an area of active research, and here we aim to provide the most practical solutions for the majority of deep learning use cases.

For an overview of the library, see the Databricks [blog post](https://databricks.com/blog/2017/06/06/databricks-vision-simplify-large-scale-deep-learning.html?preview=true) introducing Deep Learning Pipelines. For the various use cases the package serves, see the [Quick user guide](#quick-user-guide) section below.

The library is in its early days, and we welcome everyone's feedback and contribution.

Maintainers: Bago Amirbekian, Joseph Bradley, Yogesh Garg, Sue Ann Hong, Tim Hunter, Siddharth Murching, Tomas Nykodym, Lu Wang

## Spark version compatibility

To work with the latest code, Spark 2.4.0 is required and Python 3.6 & Scala 2.12 are recommended . See the [travis config](https://github.com/databricks/spark-deep-learning/blob/master/.travis.yml) for the regularly-tested combinations.

Compatibility requirements for each release are listed in the [Releases](#releases) section.


## Support

You can ask questions and join the development discussion on the [DL Pipelines Google group](https://groups.google.com/forum/#!forum/dl-pipelines-users/).

You can also post bug reports and feature requests in Github issues.


## Releases
Visit [Github Release Page](https://github.com/databricks/spark-deep-learning/releases) to check the release notes.


## License
* The Deep Learning Pipelines source code is released under the Apache License 2.0 (see the LICENSE file).
