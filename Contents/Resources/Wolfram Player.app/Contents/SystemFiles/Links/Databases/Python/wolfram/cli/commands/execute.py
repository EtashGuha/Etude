# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from wolfram.cli.dispatch import SimpleCommand
from wolfram.execution.evaluator import WolframLanguageEvaluator


class Command(SimpleCommand):
    def add_arguments(self, parser):
        parser.add_argument("data")
        parser.add_argument(
            '--kernel', dest='kernel', default=False, action='store_true')

    def handle(self, data, kernel=False):

        with WolframLanguageEvaluator() as evaluator:
            if kernel:
                self.print(evaluator.safe_evaluate_from_string(data))
            else:
                self.print(evaluator.evaluate_from_string(data))
