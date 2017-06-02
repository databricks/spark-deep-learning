all: 2.1.0s2.10 2.0.2 2.1.0

clean:
	rm -rf target/sparkdl_*.zip

2.0.2 2.1.0:
	build/sbt -Dspark.version=$@ spDist

2.1.0s2.10:
	build/sbt -Dspark.version=2.1.0 -Dscala.version=2.10.6 spDist assembly test
