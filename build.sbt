// Your sbt build file. Guides on how to write one can be found at
// http://www.scala-sbt.org/0.13/docs/index.html

import ReleaseTransformations._

val sparkVer = sys.props.getOrElse("spark.version", "2.4.0")
val sparkBranch = sparkVer.substring(0, 3)
val defaultScalaVer = sparkBranch match {
  case "2.3" => "2.11.8"
  case "2.4" => "2.11.8"
  case _ => throw new IllegalArgumentException(s"Unsupported Spark version: $sparkVer.")
}
val scalaVer = sys.props.getOrElse("scala.version", defaultScalaVer)
val scalaMajorVersion = scalaVer.substring(0, scalaVer.indexOf(".", scalaVer.indexOf(".") + 1))

sparkVersion := sparkVer

scalaVersion := scalaVer

name := "spark-deep-learning"

spName := "databricks/spark-deep-learning"

organization := "com.databricks"

version := (version in ThisBuild).value + s"-spark$sparkBranch"

// All Spark Packages need a license
licenses := Seq("Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0"))

isSnapshot := version.value.contains("-SNAPSHOT")

spAppendScalaVersion := true

// Add Spark components this package depends on, e.g, "mllib", ....
sparkComponents ++= Seq("mllib-local", "mllib", "sql")

// uncomment and change the value below to change the directory where your zip artifact will be created
// spDistDirectory := target.value

// add any Spark Package dependencies using spDependencies.
// e.g. spDependencies += "databricks/spark-avro:0.1"
spDependencies += s"databricks/tensorframes:0.6.0-s_$scalaMajorVersion"


libraryDependencies ++= Seq(
  // Update to scala-logging 3.9.0 after we update TensorFrames.
  "com.typesafe.scala-logging" %% "scala-logging-api" % "2.1.2",
  "com.typesafe.scala-logging" %% "scala-logging-slf4j" % "2.1.2",
  // Matching scalatest versions from TensorFrames
  "org.scalactic" %% "scalactic" % "3.0.0" % "test",
  "org.scalatest" %% "scalatest" % "3.0.0" % "test"
)

assemblyMergeStrategy in assembly := {
  case "requirements.txt" => MergeStrategy.concat
  case "LICENSE-2.0.txt" => MergeStrategy.rename
  case x =>
    val oldStrategy = (assemblyMergeStrategy in assembly).value
    oldStrategy(x)
}

parallelExecution := false

scalacOptions in (Compile, doc) ++= Seq(
  "-groups",
  "-implicits",
  "-skip-packages", Seq("org.apache.spark").mkString(":"))

scalacOptions in (Test, doc) ++= Seq("-groups", "-implicits")

// This fixes a class loader problem with scala.Tuple2 class, scala-2.11, Spark 2.x
fork in Test := true

// This and the next line fix a problem with forked run: https://github.com/scalatest/scalatest/issues/770
javaOptions in Test ++= Seq(
  "-Xmx2048m",
  "-XX:ReservedCodeCacheSize=384m",
  "-XX:MaxPermSize=384m",
  "-Djava.awt.headless=true"
)

concurrentRestrictions in Global := Seq(
  Tags.limitAll(1))

autoAPIMappings := true

coverageHighlighting := false

unmanagedResources in Compile += baseDirectory.value / "LICENSE"

unmanagedResourceDirectories in Compile += baseDirectory.value / "python"

includeFilter in unmanagedResources := "requirements.txt" ||
   new SimpleFileFilter(
     _.relativeTo(baseDirectory.value / "python")
       .forall(_.getPath.matches("^sparkdl/.*\\.py$")))

// Reset mappings in spPackage to avoid including duplicate files.
mappings in (Compile, spPackage) := (mappings in (Compile, packageBin)).value

// We only use sbt-release to update version numbers for now.
releaseProcess := Seq[ReleaseStep](
  inquireVersions,
  setReleaseVersion,
  commitReleaseVersion,
  tagRelease,
  setNextVersion,
  commitNextVersion
)

// Skip tests during assembly
test in assembly := {}

assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)
