#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

require 'fileutils'
include FileUtils

if not (ENV['SKIP_API'] == '1')
  if not (ENV['SKIP_SCALADOC'] == '1')
    # Build Scaladoc for Java/Scala

    puts "Moving to project root and building API docs."
    curr_dir = pwd
    cd("..")

    puts "Running 'build/sbt clean compile doc' from " + pwd + "; this may take a few minutes..."
    system("build/sbt clean compile doc") || raise("Doc generation failed")

    puts "Moving back into docs dir."
    cd("docs")

    puts "Removing old docs"
    puts `rm -rf api`

    # Copy over the unified ScalaDoc for all projects to api/scala.
    # This directory will be copied over to _site when `jekyll` command is run.
    source = "../target/scala-2.11/api"
    dest = "api/scala"

    puts "Making directory " + dest
    mkdir_p dest

    # From the rubydoc: cp_r('src', 'dest') makes src/dest, but this doesn't.
    puts "cp -r " + source + "/. " + dest
    cp_r(source + "/.", dest)

    # Append custom JavaScript
    js = File.readlines("./js/api-docs.js")
    js_file = dest + "/lib/template.js"
    File.open(js_file, 'a') { |f| f.write("\n" + js.join()) }

    # Append custom CSS
    css = File.readlines("./css/api-docs.css")
    css_file = dest + "/lib/template.css"
    File.open(css_file, 'a') { |f| f.write("\n" + css.join()) }
  end

  if not (ENV['SKIP_PYTHONDOC'] == '1')
    # Build Sphinx docs for Python

    # Get and set release version
    version = File.foreach('_config.yml').grep(/^SPARKDL_VERSION: (.+)$/){$1}.first
    version ||= 'Unknown'

    puts "Moving to python/docs directory and building sphinx."
    cd("../python/docs")
    if not (ENV['SPARK_HOME'])
      raise("Python API docs cannot be generated if SPARK_HOME is not set.")
    end
    system({"PACKAGE_VERSION"=>version}, "make clean") || raise("Python doc clean failed")
    system({"PACKAGE_VERSION"=>version}, "make html") || raise("Python doc generation failed")

    puts "Moving back into home dir."
    cd("../../")

    puts "Making directory api/python"
    mkdir_p "docs/api/python"

    puts "cp -r python/docs/_build/html/. docs/api/python"
    cp_r("python/docs/_build/html/.", "docs/api/python")
  end
end
