# Copyright 2017 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# pylint: disable=no-self-use
# pylint: disable=too-few-public-methods

from __future__ import absolute_import, division, print_function

import logging

class HorovodRunner(object):
    """
    HorovodRunner runs distributed deep learning training jobs using Horovod.

    On Databricks Runtime 5.0 ML and above, it launches the Horovod job as a distributed Spark job.
    It makes running Horovod easy on Databricks by managing the cluster setup and integrating with
    Spark.
    Check out Databricks documentation to view end-to-end examples and performance tuning tips.

    The open-source version only runs the job locally inside the same Python process,
    which is for local development only.

    .. note:: Horovod is a distributed training framework developed by Uber.
    """

    # pylint: disable=invalid-name
    def __init__(self, *, np, driver_log_verbosity="log_callback_only"):
        """
        :param np: number of parallel processes to use for the Horovod job.
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
        :param driver_log_verbosity: driver log verbosity, "all" or "log_callback_only"(default).
            During training, the first worker process will collect logs from all workers.
            The training logs are always merged into the first Spark executors stderr logs.
            If driver log verbosity is "all", HorovodRunner streams all logs to the driver and shows
            them in the notebook cell output.
            However, this might generate excessive amount of logs during distributed training.
            You can turn it off by setting driver log verbosity to "log_callback_only".
            In this mode, HorovodRunner will only stream selected logs from remote worker.
            logs from remote worker can be sent by :func:`sparkdl.horovod.log_to_driver`, or use
            log callback in the first worker process, e.g.,
            :class:`sparkdl.horovod.tensorflow.keras.LogCallback`.
        """
        self.num_processor = np

    def run(self, main, **kwargs):
        """
        Runs a Horovod training job invoking main(**kwargs).

        The open-source version only invokes main(**kwargs) inside the same Python process.
        On Databricks Runtime 5.0 ML and above, it will launch the Horovod job based on the
        documented behavior of `np`.  Both the main function and the keyword arguments are
        serialized using cloudpickle and distributed to cluster workers.

        :param main: a Python function that contains the Horovod training code.
            The expected signature is `def main(**kwargs)` or compatible forms.
            Because the function gets pickled and distributed to workers,
            please change global states inside the function, e.g., setting logging level,
            and be aware of pickling limitations.
            Avoid referencing large objects in the function, which might result large pickled data,
            making the job slow to start.
        :param kwargs: keyword arguments passed to the main function at invocation time.
        :return: return value of the main function.
            With `np>=0`, this returns the value from the rank 0 process. Note that the returned
            value should be serializable using cloudpickle.
        """
        logger = logging.getLogger("HorovodRunner")
        logger.warning(
            "You are running the open-source version of HorovodRunner. "
            "It only does basic checks and invokes the main function, "
            "which is for local development only. "
            "Please use Databricks Runtime ML 5.0+ to distribute the job.")
        return main(**kwargs)
