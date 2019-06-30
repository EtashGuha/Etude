# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import os
import tempfile
import uuid

from wolfram.cli.dispatch import SimpleCommand
from wolfram.execution.evaluator import WolframLanguageEvaluator
from wolframclient.language import wl


class Command(SimpleCommand):
    def add_arguments(self, parser):
        parser.add_argument('connection')
        parser.add_argument('--output', dest='output', default=None)

    def handle(self, connection, output=False, encoding=None, tables=None):

        with WolframLanguageEvaluator() as evaluator:

            path = output or os.path.join(tempfile.gettempdir(),
                                          'dump-%s.wl' % uuid.uuid1())

            result = evaluator.evaluate(
                wl.WithEnvironment(
                    wl.DatabaseDump(tables), {
                        'path': None,
                        "connection": connection,
                        "metadata": wl.Automatic
                    }))

            self.print(result)
