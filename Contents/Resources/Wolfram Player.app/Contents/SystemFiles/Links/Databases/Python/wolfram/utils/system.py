# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import os


def collect_python_files(*paths):
    for root in paths:
        for folder, dirs, files in os.walk(root):
            for file in files:
                if file.endswith('.py'):
                    yield os.path.join(root, folder, file)
