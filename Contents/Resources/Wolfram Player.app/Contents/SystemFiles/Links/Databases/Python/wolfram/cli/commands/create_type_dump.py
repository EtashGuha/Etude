# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import os

from wolfram.cli.dispatch import SimpleCommand
from wolfram.sql.types import database_types
from wolframclient.serializers import export
from wolframclient.utils.decorators import to_dict


class Command(SimpleCommand):
    @to_dict
    def generate_database_conversion(self):
        return database_types

    def handle(self, path=None, target_format='wxf', verbosity=False):

        path = path or os.path.normpath(
            os.path.join(
                os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
                os.pardir, 'Data', 'Types.%s' % target_format))

        if verbosity:
            print(export(self.generate_database_conversion()))

        export(
            self.generate_database_conversion(),
            path,
            target_format=target_format)

        self.print(path)
