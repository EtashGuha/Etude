# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from wolfram.cli.dispatch import SimpleCommand
from wolfram.sql import sqlalchemy as sa
from wolfram.sql.types import database_types as types
from wolframclient.utils.decorators import decorate, to_dict
from wolframclient.utils.functional import first


class Command(SimpleCommand):
    @decorate(frozenset)
    def get_available_types(self, dialect):

        module = sa.databases[dialect]

        for value in module.dialect.ischema_names.values():
            yield value

        for var in dir(module):
            value = getattr(module, var)
            if isinstance(value, type) and issubclass(value,
                                                      sa.types.TypeEngine):
                yield value

    @to_dict
    def collect_dialect_info(self, dialect):

        for value in self.get_available_types(dialect):

            match = first(
                filter(lambda t: t.__name__ in types.wolfram, value.__mro__))

            if match:
                yield value.__name__.upper(), match.__name__

    def handle(self):

        for dialect in sa.databases.keys():

            print(dialect, self.collect_dialect_info(dialect))
