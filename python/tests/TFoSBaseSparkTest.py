import sys
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

if sys.version_info[:2] <= (2, 6):
    try:
        import unittest2 as unittest
    except ImportError:
        sys.stderr.write('Please install unittest2 to test with Python 2.6 or earlier')
        sys.exit(1)
else:
    import unittest


class TFoSBaseSparkTest(unittest.TestCase):
    """Base class for unittests using Spark.  Sets up and tears down a cluster per test class"""

    @classmethod
    def setUpClass(cls):
        import os
        master = os.getenv('MASTER')
        assert master is not None, "Please start a Spark standalone cluster and export MASTER to your env."

        num_workers = os.getenv('SPARK_WORKER_INSTANCES')
        assert num_workers is not None, "Please export SPARK_WORKER_INSTANCES to your env."
        cls.num_workers = int(num_workers)

        spark_jars = os.getenv('SPARK_CLASSPATH')
        assert spark_jars and 'tensorflow-hadoop' in spark_jars, "Please add path to tensorflow-hadoop-*.jar to SPARK_CLASSPATH."

        cls.conf = SparkConf().set('spark.jars', spark_jars)
        cls.sc = SparkContext(master, cls.__name__, conf=cls.conf)
        cls.spark = SparkSession.builder.getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
        cls.sc.stop()


if __name__ == '__main__':
    unittest.main()
