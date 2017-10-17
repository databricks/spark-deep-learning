package libdeps

/**
 ======================================================
 * Build parameters
 ======================================================
 */
object LibVers {

  lazy val sparkVer = sys.props.getOrElse("spark.version", "2.2.0")
  lazy val sparkBranch = sparkVer.substring(0, 3)
  lazy val defaultScalaVer = sparkBranch match {
    case "2.0" => "2.11.8"
    case "2.1" => "2.11.8"
    case "2.2" => "2.11.8"
    case _ => throw new IllegalArgumentException(s"Unsupported Spark version: $sparkVer.")
  }

  lazy val scalaVer = sys.props.getOrElse("scala.version", defaultScalaVer)
  lazy val scalaMajorVer = scalaVer.substring(0, scalaVer.indexOf(".", scalaVer.indexOf(".") + 1))

  lazy val defaultScalaTestVer = scalaVer match {
    case s if s.startsWith("2.10") => "2.0"
    case s if s.startsWith("2.11") => "2.2.6" // scalatest_2.11 does not have 2.0 published
  }
}
