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

import java.io.{File, FileOutputStream, IOException}
import java.net.{SocketTimeoutException, URL, URLConnection}
import java.nio.channels.Channels
import java.nio.file.{Files, Path, Paths}
import java.security.{DigestInputStream, MessageDigest}
import java.util.Base64

import org.apache.commons.io.IOUtils
import org.tensorflow.framework.GraphDef
import org.tensorframes.impl.{SerializedGraph, TensorFlowOps}

private[sparkdl] class ModelFetchHelper(val cacheDirectory: String) {

  private val defaultTimeout = 10000

  private[sparkdl] class ModelFetchException(message: String) extends Exception(message)

  /**
   * Returns the absolute path of a directory in the user's home directory where we can download
   * models. If the directory does not already exist, create the directory.
   */
  private[sparkdl] def getOrCreateModelCacheDir: String = {
    val home = System.getProperty("user.home")
    val modelDir = new File(home, cacheDirectory)
    modelDir.mkdirs()
    modelDir.getAbsolutePath()
  }

  /**
   * Import GraphDef from file and check the checksum matches.
   *
   * @param filePath Path of graph file.
   * @param base64Hash Checksum to validate file.
   * @return GraphDef if checksum matches or None.
   */
  private[sparkdl] def importGraph(filePath: Path, base64Hash: String): Option[GraphDef] = {
    var bytes: Array[Byte] = null
    val messageDigest = MessageDigest.getInstance("SHA-256")
    val inputStream = Files.newInputStream(filePath)
    val digestInputStream = new DigestInputStream(inputStream, messageDigest)
    try {
      bytes = IOUtils.toByteArray(digestInputStream)
    } finally {
      inputStream.close()
      digestInputStream.close()
    }
    val digest = Base64.getEncoder.encodeToString(messageDigest.digest)
    if (digest == base64Hash) {
      Some(TensorFlowOps.readGraphSerial(SerializedGraph.create(bytes)))
    } else {
      None
    }
  }

  /**
   * Get GraphDef from URL. If the file is already in the cache directory and the checksum matches,
   * we'll skip the download. Otherwise the file will be downloaded to the cache directory. If
   * the checksum does not match after download, raise a ModelFetchException.
   *
   * @param source url source for model.
   * @param fileName file path where we will look for or cache the model file.
   * @param base64Hash check sum to verify file contents.
   * @return model as a GraphDef.
   */
  private[sparkdl] def getFromWeb(
    source: String,
    fileName: String,
    base64Hash: String): GraphDef = {

    val filePath = Paths.get(getOrCreateModelCacheDir, fileName)
    val graph = if (Files.exists(filePath)) {
      importGraph(filePath, base64Hash)
    } else {
      None
    }
    graph.getOrElse {
      // Graph file did not exist or had the wrong hash, try downloading from source.
      var urlConnection: URLConnection = null
      try {
        urlConnection = new URL(source).openConnection()
        urlConnection.setConnectTimeout(defaultTimeout)
      } catch {
        case (_: IOException | _: SocketTimeoutException) =>
          throw new IOException(s"Could not connect to $source to download model.")
      }
      val urlChannel = Channels.newChannel(urlConnection.getInputStream)
      val fileChannel = new FileOutputStream(filePath.toFile).getChannel
      try {
        fileChannel.transferFrom(urlChannel, 0, Long.MaxValue)
      } finally {
        fileChannel.close()
        urlChannel.close()
      }
      importGraph(filePath, base64Hash).getOrElse {
        // Graph still has wrong hash after download.
        throw new ModelFetchException(
          s"""Model downloaded from $source does not have the expected checksum: $base64Hash."""
        )
      }
    }
  }
}

object ModelFetcher extends ModelFetchHelper(".spark-deep-learning")
