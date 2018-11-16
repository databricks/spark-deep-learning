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
# Reference: https://hub.docker.com/r/continuumio/miniconda/~/dockerfile/
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Create sparkdl conda env.
ENV PYTHON_VERSION $PYTHON_VERSION
COPY ./environment.yml /tmp/environment.yml
RUN /opt/conda/bin/conda create -n sparkdl python=$PYTHON_VERSION && \
    /opt/conda/bin/conda env update -n sparkdl -f /tmp/environment.yml && \
    echo "conda activate sparkdl" >> ~/.bashrc

# Install Spark and update env variables.
ENV SCALA_VERSION 2.11.8
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
ENV PYSPARK_PYTHON python

# Set env variables for tests.
ENV RUN_ONLY_LIGHT_TESTS True

# The sparkdl dir will be mmounted here.
VOLUME /mnt/sparkdl
WORKDIR /mnt/sparkdl

ENTRYPOINT service ssh restart && /bin/bash
