Deep Learning Pipelines for Apache Spark
============================================================
[![Build Status][pkg-build-badge]][pkg-build-link] [![Coverage][pkg-cov-badge]][pkg-cov-link]

  [pkg-build-badge]: https://travis-ci.org/databricks/spark-deep-learning.svg?branch=master
  [pkg-build-link]: https://travis-ci.org/databricks/spark-deep-learning
  [pkg-cov-badge]: https://codecov.io/gh/databricks/spark-deep-learning/coverage.svg?branch=master
  [pkg-cov-link]: https://codecov.io/gh/databricks/spark-deep-learning/branch/master

The repo only contains HorovodRunner code for local CI and API docs. To use HorovodRunner for distributed training, please use Databricks Runtime for Machine Learning,
Visit databricks doc [HorovodRunner: distributed deep learning with Horovod](https://docs.databricks.com/applications/machine-learning/train-model/distributed-training/horovod-runner.html) for details.

To use the previous release that contains Spark Deep Learning Pipelines API, please go to [Spark Packages page](https://spark-packages.org/package/databricks/spark-deep-learning).


## API Documentation

### class sparkdl.HorovodRunner(\*, np, driver_log_verbosity='all')
Bases: `object`

HorovodRunner runs distributed deep learning training jobs using Horovod.

On Databricks Runtime 5.0 ML and above, it launches the Horovod job as a distributed Spark job.
It makes running Horovod easy on Databricks by managing the cluster setup and integrating with
Spark.
Check out Databricks documentation to view end-to-end examples and performance tuning tips.

The open-source version only runs the job locally inside the same Python process,
which is for local development only.

**NOTE**: Horovod is a distributed training framework developed by Uber.


#### run(main, \*\*kwargs)
Runs a Horovod training job invoking main(\*\*kwargs).

The open-source version only invokes main(\*\*kwargs) inside the same Python process.
On Databricks Runtime 5.0 ML and above, it will launch the Horovod job based on the
documented behavior of np.  Both the main function and the keyword arguments are
serialized using cloudpickle and distributed to cluster workers.


* **Parameters**

    
    * **main** – a Python function that contains the Horovod training code.
    The expected signature is def main(\*\*kwargs) or compatible forms.
    Because the function gets pickled and distributed to workers,
    please change global states inside the function, e.g., setting logging level,
    and be aware of pickling limitations.
    Avoid referencing large objects in the function, which might result large pickled data,
    making the job slow to start.


    * **kwargs** – keyword arguments passed to the main function at invocation time.



* **Returns**

    return value of the main function.
    With np>=0, this returns the value from the rank 0 process. Note that the returned
    value should be serializable using cloudpickle.


## Releases
Visit [Github Release Page](https://github.com/databricks/spark-deep-learning/releases) to check the release notes.


## License
* The source code is released under the Apache License 2.0 (see the LICENSE file).
