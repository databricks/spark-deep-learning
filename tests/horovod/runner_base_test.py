# Copyright 2017 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import absolute_import, division, print_function

from inspect import getfullargspec, FullArgSpec
import unittest

from sparkdl import HorovodRunner


class HorovodRunnerBaseTestCase(unittest.TestCase):

    def test_func_signature(self):
        """Test that __init__ and run signatures are correct."""
        init_spec = getfullargspec(HorovodRunner.__init__)
        self.assertEquals(init_spec, FullArgSpec(
            args=['self'], varargs=None, varkw=None, defaults=None,
            kwonlyargs=['np', 'driver_log_verbosity'],
            kwonlydefaults={'driver_log_verbosity': 'log_callback_only'}, annotations={}))
        run_spec = getfullargspec(HorovodRunner.run)
        self.assertEquals(run_spec, FullArgSpec(
            args=['self', 'main'], varargs=None, varkw='kwargs', defaults=None, kwonlyargs=[],
            kwonlydefaults=None, annotations={}
        ))

    def test_init_keyword_only(self):
        """Test that user must use keyword args in __init__"""
        with self.assertRaises(TypeError):
            HorovodRunner(2)

    def test_run(self):
        """Test that run just invokes the main method in the same process."""
        hr = HorovodRunner(np=-1)
        data = []

        def append(value):
            data.append(value)

        hr.run(append, value=1)
        self.assertEquals(data[0], 1)

    def test_return_value(self):
        """Test that the return value is returned to the user."""
        hr = HorovodRunner(np=-1)
        return_value = hr.run(lambda: 42)
        self.assertEquals(return_value, 42)
