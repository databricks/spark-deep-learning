// You may use this file to add plugin dependencies for sbt.

// resolvers += "Local Spark repo" at "file:///Users/tomas/.m2/repository" // add local spark repo
resolvers += "Spark Packages repo" at "https://dl.bintray.com/spark-packages/maven/"

addSbtPlugin("org.spark-packages" %% "sbt-spark-package" % "0.2.5")

// scalacOptions in (Compile,doc) := Seq("-groups", "-implicits")

addSbtPlugin("org.scoverage" % "sbt-scoverage" % "1.5.0")
