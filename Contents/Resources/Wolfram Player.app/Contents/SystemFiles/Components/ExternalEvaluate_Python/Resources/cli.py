# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import sys
import traceback
import argparse
import logging

def setup_verbose_logging(filepath):
    logging.basicConfig(
        filename=filepath,
        filemode='a',
        format='%(asctime)s, %(name)s %(levelname)s %(message)s',
        level=logging.DEBUG)


def create_error_logger():
    ee_logger = logging.getLogger(name='ExternalEvaluate')

    def logfunction(exctype, value, tb):
        for msg in traceback.format_exception(exctype, value, tb):
            ee_logger.error(msg)

    return logfunction



#this is used by external evaluate to start the actual loop
def execute_from_command_line():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required = False, action = 'append', default = [])
    parser.add_argument('--excepthook', required = False)

    args, extra = parser.parse_known_args()

    for p in args.path:
        sys.path.insert(0, p)

    # Setup the except hook as soon as possible to log as many errors as possible.
    # Especially some occurring while loading the client library.
    if args.excepthook:
        sys.excepthook = create_error_logger()
        setup_verbose_logging(args.excepthook)

    from wolframclient.cli.dispatch import execute_from_command_line as _execute

    _execute([sys.argv[0]] + extra)

if __name__ == '__main__':

    execute_from_command_line()
