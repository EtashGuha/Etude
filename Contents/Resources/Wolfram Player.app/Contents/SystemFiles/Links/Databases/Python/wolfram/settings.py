# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import os

from wolframclient.utils.datastructures import Settings

settings = Settings(
    DEBUG = bool(os.environ.get('WOLFRAM_SQL_DEBUG_MODE', False)),
    TARGET_KERNEL_VERSION = float(os.environ.get('WOLFRAM_KERNEL_VERSION', 12)),
)