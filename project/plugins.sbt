// You may use this file to add plugin dependencies for sbt.
resolvers += "Spark Packages repo" at "https://dl.bintray.com/spark-packages/maven/"
addSbtPlugin("org.spark-packages" %% "sbt-spark-package" % "0.2.5")
// scalacOptions in (Compile,doc) := Seq("-groups", "-implicits")
addSbtPlugin("org.scoverage" % "sbt-scoverage" % "1.5.0")

addSbtPlugin("com.github.gseitz" % "sbt-release" % "1.0.8")
