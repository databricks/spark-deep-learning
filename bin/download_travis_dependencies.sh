#!/usr/bin/env bash
echo "Downloading Spark if necessary"
echo "Spark version = $SPARK_VERSION"
echo "Spark build = $SPARK_BUILD"
echo "Spark build URL = $SPARK_BUILD_URL"
mkdir -p $HOME/.cache/spark-versions
filename="$HOME/.cache/spark-versions/$SPARK_BUILD.tgz"
if ! [ -f $filename ]; then
	echo "Downloading file..."
	wget "$SPARK_BUILD_URL" -O $filename
	echo "[Debug] Following should list a valid spark binary"
	ls -larth $HOME/.cache/spark-versions/*
	tar -xvzf $filename --directory $HOME/.cache/spark-versions > /dev/null
fi
