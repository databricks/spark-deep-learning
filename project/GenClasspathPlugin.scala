package sbtgenclasspath

import sbt._, Keys._
import sbtsparkpackage.SparkPackagePlugin.autoImport._
import libdeps.LibVers._

object GenClasspathPlugin extends sbt.AutoPlugin {

  object autoImport {

    lazy val genClasspath = taskKey[Unit]("Build runnable script with classpath")
    lazy val extraSparkSubmitModules = settingKey[Seq[ModuleID]]("Additional spark submit jar dependencies")

    lazy val genClasspathSettings: Seq[Def.Setting[_]] = Seq(

      extraSparkSubmitModules := Seq.empty[ModuleID],

      genClasspath := {
        import java.io.PrintWriter

        val sbtPathRoot = baseDirectory.value / ".sbt.paths"
        sbtPathRoot.mkdirs()

        def writeClasspath(cpType: String)(R: => String): Unit = {
          val fout = new PrintWriter((sbtPathRoot / s"SBT_${cpType}_CLASSPATH").toString)
          println(s"Building ${cpType} classpath for current project")
          try fout.write(R) finally fout.close()
        }

        writeClasspath("RUNTIME") {
          (fullClasspath in Runtime).value.files.map(_.toString).mkString(":")
        }

        writeClasspath("SPARK_PACKAGE") {
          import scala.util.matching.Regex
          val patt = s"(.+?)/(.+?):(.+?)(-s_${scalaMajorVer})?".r
          val pkgs = (spDependencies.value).map { _ match {
            case patt(orgName, pkgName, pkgVer, stem, _*) =>
              if (null != stem) {
                println(s"org ${orgName}, pkg ${pkgName}, ver ${pkgVer}, ${stem}")
                s"${pkgName}-${pkgVer}${stem}.jar"
              } else {
                println(s"org ${orgName}, pkg ${pkgName}, ver ${pkgVer}")
                s"${pkgName}-${pkgVer}.jar"
              }
          }}.toSet

          // TODO: not knowing the proper way, I just fall back to Regex
          val extraSpModIds = (extraSparkSubmitModules in Compile).value.flatMap { mod =>
            //"com.typesafe.scala-logging:scala-logging-api:2.1.2"
            // scala-logging-api_2.11-2.1.2.jar
            val patt = s"(.+?):(.+?):(.+?)".r
            mod.toString match {
              case patt(orgName, pkgName, pkgVer) =>
                Seq(s"${pkgName}_${scalaMajorVer}-${pkgVer}.jar", s"${pkgName}-${pkgVer}.jar")
            }
          }.toSet

          (fullClasspath in Compile).value.files.filter { cpFile =>
            val cpName = cpFile.getName
            println(cpName)
            (pkgs contains cpName) || (extraSpModIds contains cpName)
          }.map(_.toString).mkString(":")
        }
      }
    )
  }
  import autoImport._

  override def requires = sbt.plugins.JvmPlugin

  // This plugin is automatically enabled for projects which are JvmPlugin.
  override def trigger = allRequirements

  // a group of settings that are automatically added to projects.
  override val projectSettings =
    inConfig(Compile)(genClasspathSettings) ++ inConfig(Test)(genClasspathSettings)
}
