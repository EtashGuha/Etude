# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import string
from decimal import ROUND_HALF_EVEN, Decimal

from wolfram.sql import sqlalchemy as sa
from wolfram.sql.types import database_types as types
from wolfram.tests.utils.base import TestCase as BaseTestCase

NUM = 1.1
STR = string.ascii_letters


def quantize(n=0, decimals=4, rounding=ROUND_HALF_EVEN):
    return Decimal(n).quantize(
        Decimal(1) / 10**decimals, rounding=ROUND_HALF_EVEN)


class TestCase(BaseTestCase):
    def create_db_for_inspection(self, metadata):
        sa.Table(
            'test',
            metadata,
            sa.Column('integer', types.wolfram.Integer(), primary_key=True),
            sa.Column('real', types.wolfram.Real()),
            sa.Column('decimal', types.wolfram.Decimal(10, 4)),
            sa.Column('string', types.wolfram.String(len(STR))),
        )

    def test_inspection(self):

        for metadata in self.create_test_databases(
                self.create_db_for_inspection):

            table = metadata.tables['test']

            with metadata.bind.connect() as connection:

                connection.execute(
                    table.insert(), {
                        'integer': int(NUM),
                        'real': float(NUM),
                        'decimal': quantize(NUM),
                        'string': STR
                    })

                select = lambda *args: tuple(connection.execute(sa.select(args)))
                cast = lambda e, t: sa.cast(e, t, dialect=metadata.bind.engine)

                for column, value in (
                    (table.c.integer, int(NUM)),
                    (table.c.real, float(NUM)),
                    (table.c.decimal, quantize(NUM)),
                ):
                    self.assertEqual(
                        select(
                            cast(column, types.wolfram.Integer()),
                            cast(column, types.wolfram.Real()),
                            cast(column, types.wolfram.Decimal(10, 4)),
                        ),
                        ((int(value), float(value), quantize(value)), ),
                    )

                #all kinds of inconsistent behaviour are happening when casting to string
                #
                #self.assertEqual(
                #    select(
                #        cast(table.c.integer, types.wolfram.String(20)),
                #        cast(table.c.real,    types.wolfram.String(20)),
                #        cast(table.c.decimal, types.wolfram.String(20)),
                #    ), (
                #        ('1', '1.0', '1.0000'),
                #    ),
                #)

                #SQLITE is NOT truncating the string
                #
                #self.assertEqual(
                #    select(
                #        cast(table.c.string, types.wolfram.String(10)),
                #        cast(table.c.string, types.wolfram.String(20)),
                #    ), (
                #        (STR, STR),
                #    ),
                #)
