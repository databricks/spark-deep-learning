/*
 * Copyright 2017 Databricks, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.databricks.sparkdl

import java.io.{File, FileOutputStream, FileWriter, OutputStream}
import java.nio.file.{Files, Paths}

import org.apache.commons.io.FileUtils
import org.scalatest.FunSuite

class ModelFetcherSuite extends FunSuite {
  val testGraphDefFile = getClass.getResource("/sparkdl/test_net.pb").getFile
  val testGraphHash = "jVCEKp1bV53eib8d8OKreTH4fHu/Ji5NHMOsgdVwbMg="

  test("Import graph loads a graph only if the hash matches.") {
    assert(ModelFetcher.importGraph(Paths.get(testGraphDefFile), testGraphHash).isDefined)
    assert(ModelFetcher.importGraph(Paths.get(testGraphDefFile), "badHash").isEmpty)
  }

  test("Test getFromWeb using a local url.") {
    // Make a test version of ModelFetchHelper which uses a different test directory.
    val testModelFetcher = new ModelFetchHelper(".spark-deep-learning-test")
    val cacheDir = testModelFetcher.getOrCreateModelCacheDir
    // Delete the cache directory to test in a clean environment.
    FileUtils.deleteDirectory(new File(cacheDir))
    val cacheFile = "test_net_cached.pb"

    val gdef1 = testModelFetcher.getFromWeb(
      "file:" + testGraphDefFile,
      cacheFile,
      testGraphHash)
    assert(gdef1 === Models.TestNet.graph)
    assert(Files.exists(Paths.get(cacheDir, cacheFile)))

    // If we modify the file, the hash should no longer match.
    val f = new FileWriter(cacheFile, true)
    f.write("hash should change.")
    f.close()
    assert(testModelFetcher.importGraph(Paths.get(cacheFile), testGraphHash).isEmpty)

    // If the cache file has a bad hash, get from web should replace it.
    val gdef2 = testModelFetcher.getFromWeb(
      "file:" + testGraphDefFile,
      cacheFile,
      testGraphHash)
    assert(gdef2 === Models.TestNet.graph)
    assert(Files.exists(Paths.get(cacheDir, cacheFile)))

    // This should try and replace the cache file, but fail when the replacement has the wrong hash.
    assertThrows[testModelFetcher.ModelFetchException] {
      testModelFetcher.getFromWeb(
        "file:" + testGraphDefFile,
        cacheFile,
        "badHash")
    }
  }
}
