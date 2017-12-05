package org.apache.spark.ml.util

import java.io.File

import org.scalatest.{BeforeAndAfterAll, Suite}

import org.apache.spark.util.Utils

/**
 * Trait that creates a temporary directory before all tests and deletes it after all.
 */
trait TempDirectory extends BeforeAndAfterAll { self: Suite =>

  private var _tempDir: File = _

  /**
   * Returns the temporary directory as a `File` instance.
   */
  protected def tempDir: File = _tempDir

  override def beforeAll(): Unit = {
    super.beforeAll()
    _tempDir = Utils.createTempDir(namePrefix = this.getClass.getName)
  }

  override def afterAll(): Unit = {
    try {
      Utils.deleteRecursively(_tempDir)
    } finally {
      super.afterAll()
    }
  }
}

