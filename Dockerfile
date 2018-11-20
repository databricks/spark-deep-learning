FROM ubuntu:16.04

ARG PYTHON_VERSION=3.6

RUN apt-get update && \
    apt-get install -y wget bzip2 build-essential openjdk-8-jdk ssh sudo && \
    apt-get clean

# Add ubuntu user and enable password-less sudo
RUN useradd -mU -s /bin/bash -G sudo ubuntu && \
    echo "ubuntu ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Set up SSH. Docker will cache this layer. So the keys end up the same on all containers.
# We use a non-default location to simulate MLR, which doesn't have passwordless SSH.
# We still keep the private key for manual SSH during debugging.
RUN ssh-keygen -q -f ~/.ssh/docker -N "" && \
    cp ~/.ssh/docker.pub ~/.ssh/authorized_keys

# Install Open MPI
RUN wget --quiet https://github.com/uber/horovod/files/1596799/openmpi-3.0.0-bin.tar.gz -O /tmp/openmpi.tar.gz && \
    cd /usr/local && \
    tar -zxf /tmp/openmpi.tar.gz && \
    ldconfig && \
    rm -r /tmp/openmpi.tar.gz

# Install Miniconda.
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda
ENV PATH /opt/conda/bin:$PATH

# Install sparkdl dependencies.
# Instead of activate the "sparkdl" conda env, we update env variables directly.
ENV PYTHON_VERSION $PYTHON_VERSION
ENV PATH /opt/conda/envs/sparkdl/bin:$PATH
ENV LD_LIBRARY_PATH /opt/conda/envs/sparkdl/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH /opt/conda/envs/sparkdl/lib/python$PYTHON_VERSION/site-packages:$PYTHONPATH
COPY ./environment.yml /tmp/environment.yml
RUN conda create -n sparkdl python=$PYTHON_VERSION && \
    conda env update -n sparkdl -f /tmp/environment.yml

# Install Spark and update env variables.
ENV SPARK_VERSION 2.4.0
ENV SPARK_BUILD "spark-${SPARK_VERSION}-bin-hadoop2.7"
ENV SPARK_BUILD_URL "https://dist.apache.org/repos/dist/release/spark/spark-2.4.0/${SPARK_BUILD}.tgz"
RUN wget --quiet $SPARK_BUILD_URL -O /tmp/spark.tgz && \
    tar -C /opt -xf /tmp/spark.tgz && \
    mv /opt/$SPARK_BUILD /opt/spark && \
    rm /tmp/spark.tgz
ENV SPARK_HOME /opt/spark
ENV PATH $SPARK_HOME/bin:$PATH
ENV PYTHONPATH /opt/spark/python/lib/py4j-0.10.7-src.zip:/opt/spark/python/lib/pyspark.zip:$PYTHONPATH

# Declare env variables to run tests.
ENV SCALA_VERSION 2.11.8
ENV PYSPARK_PYTHON python
ENV RUN_ONLY_LIGHT_TESTS True

# Persist important env variables to /etc/environment.
RUN echo "SCALA_VERSION=$SCALA_VERSION" >> /etc/environment && \
    echo "SPARK_VERSION=$SPARK_VERSION" >> /etc/environment && \
    echo "SPARK_HOME=$SPARK_HOME" >> /etc/environment && \
    echo "PATH=$PATH" > /etc/environment && \
    echo "PYTHONPATH=$PYTHONPATH" >> /etc/environment && \
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> /etc/environment && \
    echo "PYSPARK_PYTHON=$PYSPARK_PYTHON" >> /etc/environment && \
    echo "RUN_ONLY_LIGHT_TESTS=$RUN_ONLY_LIGHT_TESTS" >> /etc/environment

# The sparkdl dir will be mmounted here.
VOLUME /mnt/sparkdl

ENTRYPOINT service ssh restart && /bin/bash
