FROM ubuntu:16.04

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        openjdk-8-jdk \
        openjdk-8-jre-headless \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Please don't set DEBIAN_FRONTEND.
# Checkout "Why is DEBIAN_FRONTEND=noninteractive discouraged in Dockerfiles?"
# from https://docs.docker.com/engine/faq/ for details.
# ENV DEBIAN_FRONTEND noninteractive

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        Pillow \
        h5py \
        numpy \
        pandas \
        scipy \
        sklearn \
        ipython \
        pyspark

# Install TensorFlow CPU verion from central
RUN pip --no-cache-dir install \
    'http://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-1.head-cp27-cp27mu-linux_x86_64.whl'

ENV SPARK_WORKER_PORT 8888

EXPOSE 4040 6066 7077 8080 8888

# Avoid the default Docker behavior of mapping our IP address to an unreachable host name

RUN mkdir -p /phi9t
COPY entrypoint.sh /phi9t/.

WORKDIR /workspace

ENTRYPOINT ["/phi9t/entrypoint.sh"]
