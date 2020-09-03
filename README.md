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

* **Parameters**


    * **np** - number of parallel processes to use for the Horovod job.
        This argument only takes effect on Databricks Runtime 5.0 ML and above.
        It is ignored in the open-source version.
        On Databricks, each process will take an available task slot,
        which maps to a GPU on a GPU cluster or a CPU core on a CPU cluster.
        Accepted values are:

        - If <0, this will spawn `-np` subprocesses on the driver node to run Horovod locally.
          Training stdout and stderr messages go to the notebook cell output, and are also
          available in driver logs in case the cell output is truncated. This is useful for
          debugging and we recommend testing your code under this mode first. However, be
          careful of heavy use of the Spark driver on a shared Databricks cluster.
          Note that `np < -1` is only supported on Databricks Runtime 5.5 ML and above.
        - If >0, this will launch a Spark job with `np` tasks starting all together and run the
          Horovod job on the task nodes.
          It will wait until `np` task slots are available to launch the job.
          If `np` is greater than the total number of task slots on the cluster,
          the job will fail. As of  Databricks Runtime 5.4 ML, training stdout and stderr
          messages go to the notebook cell output. In the event that the cell output is
          truncated, full logs are available in stderr stream of task 0 under the 2nd spark
          job started by HorovodRunner, which you can find in the Spark UI.
        - If 0, this will use all task slots on the cluster to launch the job.
          .. warning:: Setting np=0 is deprecated and it will be removed in the next major
            Databricks Runtime release. Choosing np based on the total task slots at runtime is
            unreliable due to dynamic executor registration. Please set the number of parallel
            processes you need explicitly.
    * **np** - driver_log_verbosity: This argument is only available on Databricks Runtime.

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
