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

import java.io.File
import javax.imageio.ImageIO

import org.apache.spark.image.ImageSchema
import org.apache.spark.sql.Row
import org.scalatest.FunSuite

object ImageUtilsSuite {
  val biggerImage: Row = {
    val biggerFile = getClass.getResource("/images/00081101.jpg").getFile
    val imageBuffer = ImageIO.read(new File(biggerFile))
    ImageUtils.spImageFromBufferedImage(imageBuffer)
  }

  val smallerImage: Row = {
    val smallerFile = getClass.getResource("/smaller.png").getFile
    val imageBuffer = ImageIO.read(new File(smallerFile))
    ImageUtils.spImageFromBufferedImage(imageBuffer)
  }

  val tgtHeight: Int = ImageSchema.getHeight(smallerImage)
  val tgtWidth: Int = ImageSchema.getWidth(smallerImage)
  val tgtChannels: Int = ImageSchema.getNChannels(smallerImage)
}

class ImageUtilsSuite extends FunSuite {
  import ImageUtilsSuite._

  test("test binary image resize") {
    val testImage = ImageUtils.resizeImage(tgtHeight, tgtWidth, tgtChannels)(biggerImage)
    assert(ImageSchema.getHeight(testImage) === tgtHeight)
    assert(ImageSchema.getWidth(testImage) === tgtWidth)
    assert(ImageSchema.getNChannels(testImage) === tgtChannels)
    val testImageData = ImageSchema.getData(testImage)
    val smallerImageData = ImageSchema.getData(smallerImage)
    assert(testImageData.deep === smallerImageData)
  }

}
