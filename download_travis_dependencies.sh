#!/usr/bin/env bash
echo "Downloading Spark if necessary"
echo "Spark version = $SPARK_VERSION"
echo "Spark build = $SPARK_BUILD"
echo "Spark build URL = $SPARK_BUILD_URL"
mkdir -p $HOME/.cache/spark-versions
filename="$HOME/.cache/spark-versions/$SPARK_BUILD.tgz"
if ! [ -f $filename ]; then
	echo "Downloading file..."
	echo `which curl`
	curl "$SPARK_BUILD_URL" > $filename
	echo "Content of directory:"
	ls -la $HOME/.cache/spark-versions/*
	tar xvf $filename --directory $HOME/.cache/spark-versions > /dev/null
fi
