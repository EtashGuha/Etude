# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from wolfram.cli.dispatch import SimpleCommand
from wolfram.execution.messages import messages
from wolframclient.language.decorators import to_wl
from wolframclient.utils.encoding import force_text


class Command(SimpleCommand):
    def add_arguments(self, parser):
        parser.add_argument('args', nargs='*')

    def handle(self, *args):
        @to_wl()
        def error():
            raise messages.invalid_expression.as_exception('a')

        self.print(force_text(error()))
