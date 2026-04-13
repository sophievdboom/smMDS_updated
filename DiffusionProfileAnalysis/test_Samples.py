# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:39:51 2017

@author: quentinpeter
"""

import os
import shutil
from unittest import TestCase


class TestSamples(TestCase):
    def test_scripts(self):
        with open("generate_generators.py") as f:
            exec(f.read())
        os.chdir("Samples")
        if os.path.exists("output"):
            shutil.rmtree("output")
        with open("generate_metadata.py") as f:
            exec(f.read())
        with open("generate_settings.py") as f:
            exec(f.read())
        with open("sizescript.py") as f:
            exec(f.read())
        self.assertTrue(os.path.exists("output"))
        shutil.rmtree("output")
