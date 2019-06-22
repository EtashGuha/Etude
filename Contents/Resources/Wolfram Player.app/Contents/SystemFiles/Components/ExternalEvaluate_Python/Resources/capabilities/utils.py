# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import itertools
import sys

DEFAULT_MODULES = ['zmq', 'traceback']


def check_compatibility(*additionalModules):
    for module in itertools.chain(DEFAULT_MODULES, additionalModules):
        try:
            __import__(module)
        except ImportError:
            return False

    vers = sys.version_info[0] + 0.1 * sys.version_info[1]
    if vers < 2.7:
        #don't support less than 2.7
        return False
    elif vers > 3.0 and vers < 3.4:
        #also don't support py3 less than 3.4
        return False
    return True


def print_compatibility(*modules):
    if check_compatibility(*modules):
        print('TRUE')
    else:
        print('FALSE')
