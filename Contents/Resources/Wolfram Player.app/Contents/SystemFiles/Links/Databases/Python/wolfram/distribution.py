# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import os
import sys

if __name__ == '__main__':

    is_distribution = getattr(sys, 'frozen', False)

    if is_distribution:
        #this is true when running the dist from pyinstaller
        current = os.path.dirname(sys.executable)
    else:
        current = os.path.dirname(os.path.realpath(__file__))

    if not current in sys.path:
        sys.path.insert(0, current)

    from wolfram.cli.dispatch import execute_from_command_line

    execute_from_command_line(distribution = is_distribution)
