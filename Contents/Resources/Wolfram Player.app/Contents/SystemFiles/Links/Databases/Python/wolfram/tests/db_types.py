# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import datetime
import decimal
from decimal import Decimal as d

from wolfram.sql import sqlalchemy as sa
from wolfram.sql.types import database_types as types
from wolfram.sql.types.mixin.base import InvalidResult
from wolfram.tests.utils.base import TestCase as BaseTestCase
from wolfram.tests.utils.dates import test_date, test_datetime, test_time
from wolfram.tests.utils.random import data_for_type
from wolfram.utils.dates import new_datetime, new_time, pytz
from wolfram.utils.normalize import dumps, serialize_type
from wolframclient.language.exceptions import WolframLanguageException
from wolframclient.settings import NOT_PROVIDED
from wolframclient.utils import six
from wolframclient.utils.encoding import force_text
from wolframclient.utils.functional import flatten, iterate


class TestCase(BaseTestCase):
    def compare_wl_form(self, a, b=None, **opts):
        return self.assertEqual(dumps(a), dumps(b), **opts)

    def compare_result_verbatim(self, types, value, comparison, **opts):

        for type in iterate(types):

            for val in value:

                self.assertEqual(
                    type.process_result_value(val, dialect=None), comparison,
                    **opts)

    def compare_result_value(self,
                             types,
                             value,
                             literal=NOT_PROVIDED,
                             instance=NOT_PROVIDED):

        for type in iterate(types):

            for val in value:

                r = type.process_result_value(val, dialect=None)

                if not literal is NOT_PROVIDED:
                    self.assertEqual(
                        r,
                        literal,
                        msg=
                        'when casting to type %s, %s should be exactly %s not %s'
                        % (type, val, literal, r))

                if not instance is NOT_PROVIDED:

                    self.assertEqual(
                        isinstance(r, instance),
                        True,
                        msg=
                        'when casting to type %s, %s should be product and instance of %s not %s'
                        % (type, val, instance, r))

    def test_type_casting(self):

        self.compare_result_value(
            types.wolfram.String(), ['just a string'],
            literal='just a string',
            instance=six.string_types)

        self.compare_result_value(
            types.wolfram.String(), [True, False, 0, 1.0],
            instance=InvalidResult)

        self.compare_result_value(
            types.wolfram.Boolean(),
            [True, 'True'],
            literal=True,
        )

        self.compare_result_value(
            types.wolfram.Boolean(),
            [False, 'False'],
            literal=False,
        )

        self.compare_result_value(
            types.wolfram.Boolean(),
            [12, 'nonsense'],
            instance=InvalidResult,
        )

        self.compare_result_value(
            types.wolfram.Integer(),
            ['1', 1, 1.0, decimal.Decimal(1.0)],
            instance=int,
            literal=1)

        self.compare_result_value(
            types.wolfram.Decimal(),
            ['1', 1, 1.0, decimal.Decimal(1.0)],
            instance=decimal.Decimal,
            literal=decimal.Decimal(1))

        self.compare_result_value(
            types.wolfram.Real(),
            ['1', 1, 1.0, decimal.Decimal(1.0)],
            instance=float,
            literal=1.0)

        self.compare_result_value(
            [
                types.wolfram.Integer(),
                types.wolfram.Decimal(),
                types.wolfram.Real()
            ],
            ['nonsense', True],
            instance=InvalidResult,
        )

        self.compare_result_value([
            types.wolfram.Integer(),
            types.wolfram.Decimal(),
            types.wolfram.Real(),
            types.wolfram.String(),
            types.wolfram.UUID(),
            types.wolfram.Choices(["a", "b", "c"]),
            types.wolfram.Date(),
            types.wolfram.DateTime(),
            types.wolfram.Time(),
        ], [None, ''],
                                  literal=None,
                                  instance=None.__class__)

        self.compare_result_value(
            types.wolfram.Date(), [
                test_date(),
                test_datetime(),
                test_datetime(tzinfo="Europe/Rome")
            ],
            literal=test_date())

        self.compare_result_value(
            types.wolfram.DateTime(True),
            [test_datetime(tzinfo="Europe/Rome")],
            literal=test_datetime(tzinfo="Europe/Rome"))

        self.compare_result_value(
            types.wolfram.DateTime(True), [test_datetime()],
            literal=test_datetime())

        self.compare_result_value(
            types.wolfram.Time(True), [
                test_time(tzinfo="Europe/Rome"),
                test_datetime(tzinfo="Europe/Rome")
            ],
            literal=test_time(tzinfo="Europe/Rome"))

        self.compare_result_value(
            types.wolfram.Time(True),
            [test_time(), test_datetime()],
            literal=test_time())

        self.compare_result_value(
            types.wolfram.DateTime(False),
            [test_datetime(tzinfo="Europe/Rome"),
             test_datetime()],
            literal=test_datetime())

        self.compare_result_value(
            types.wolfram.Time(False), [
                test_time(tzinfo="Europe/Rome"),
                test_time(),
                test_datetime(),
                test_datetime(tzinfo="Europe/Rome")
            ],
            literal=test_time())

        self.compare_result_value([
            types.wolfram.Date(),
            types.wolfram.DateTime(),
            types.wolfram.Time()
        ], [1, False],
                                  instance=InvalidResult)

    def test_arg_validation(self):

        self.compare_wl_form(types.wolfram.Boolean(), 'Boolean')
        self.compare_wl_form(types.wolfram.String(23), ['String', 23])
        self.compare_wl_form(
            types.wolfram.String(23, 'utf-8'), ['String', 23, 'utf-8'])

        with self.assertRaises(WolframLanguageException):
            self.compare_wl_form(types.wolfram.Integer('crap'))

        with self.assertRaises(WolframLanguageException):
            self.compare_wl_form(types.wolfram.String('crap'))

        self.compare_wl_form(types.mysql.BOOLEAN(), 'BOOLEAN')
        self.compare_wl_form(types.mysql.VARCHAR(23), ['VARCHAR', 23])
        self.compare_wl_form(
            types.mysql.VARCHAR(23, 'utf-8'), ['VARCHAR', 23, 'utf-8'])

        with self.assertRaises(WolframLanguageException):
            self.compare_wl_form(types.mysql.VARCHAR('crap'))

        self.compare_wl_form(sa.types.Integer(), "Integer")
        self.compare_wl_form(sa.types.Date(), "Date")

    def test_possible_specs(self):

        #ALL TYPES

        dumps(self.generate_test_types(*types.values()), indent=4)

    def generate_test_types(self, *collections):
        for collection in collections:
            for type_info, args in collection.generate_possible_tests():
                yield type_info(*args)

    def create_db_for_types(self, metadata):

        sa.Table(
            'test', metadata,
            sa.Column('id', types.wolfram.Integer(), primary_key=True),
            *(sa.Column(
                '%s_%s' % (
                    force_text(i + 1).zfill(2),
                    "_".join(
                        map(
                            force_text,
                            flatten(
                                serialize_type(
                                    type_, dialect=metadata.bind.name)))),
                ), type_)
              for i, type_ in enumerate(
                  self.generate_test_types(getattr(types, metadata.bind.name)))
              ))

    def test_insert_data(self):
        for metadata in self.create_test_databases(self.create_db_for_types):

            table = metadata.tables['test']

            with metadata.bind.connect() as connection:

                #we generate some random data using data_for_type utility

                data = [{
                    c.name: data_for_type(c.type)
                    for c in table.c if not c.primary_key
                } for i in range(10)]

                connection.execute(table.insert(), data)

                for result in connection.execute(
                        sa.select([table]).order_by('id')):
                    for value, column in zip(result, table.c):
                        #first of all we check that reading is returning the expected type
                        self.assertEqual(
                            isinstance(value,
                                       column.type.type_info.python_types),
                            True,
                            msg=
                            "Value %s is an instance of %s but it should not an instance of %s for column of type %s"
                            % (value, value.__class__,
                               column.type.type_info.python_types,
                               column.type.__class__.__name__))

    #testing numeric precision

    def create_db_for_numeric(self, metadata):

        sa.Table(
            'test_numeric',
            metadata,
            sa.Column('id', types.wolfram.Integer(), primary_key=True),
            sa.Column('integer', types.wolfram.Integer()),
            sa.Column('real', types.wolfram.Real()),
            sa.Column('decimal_2', types.wolfram.Decimal(10, 2)),
            sa.Column('decimal_4', types.wolfram.Decimal(10, 4)),
        )

    def test_numeric_insertion(self):

        for metadata in self.create_test_databases(self.create_db_for_numeric):

            with metadata.bind.connect() as connection:

                table = metadata.tables['test_numeric']

                test_data = [[1, 1, 1.2, d('1.23'),
                              d('1.2324')],
                             [2, 200, 200.123,
                              d('1.2342'),
                              d('1.23')]]

                connection.execute(table.insert(), [
                    dict(zip(table.c.keys(), test_value))
                    for test_value in test_data
                ])

                self.assertEqual(
                    [[1, 1, 1.2, d('1.23'), d('1.2324')],
                     [2, 200, 200.123, d('1.23'),
                      d('1.2300')]], [
                          list(result)
                          for result in connection.execute(sa.select([table]))
                      ])

    #testing date precision

    def create_db_for_date(self, metadata):
        sa.Table(
            'test_date',
            metadata,
            sa.Column('id', types.wolfram.Integer(), primary_key=True),
            sa.Column('date', types.wolfram.Date()),
            sa.Column('time_naive', types.wolfram.Time(False)),
            sa.Column('time_aware', types.wolfram.Time(True)),
            sa.Column('datetime_naive', types.wolfram.DateTime(False)),
            sa.Column('datetime_aware', types.wolfram.DateTime(True)),
        )

    def test_date_insertion(self):

        for metadata in self.create_test_databases(self.create_db_for_date):

            with metadata.bind.connect() as connection:

                table = metadata.tables['test_date']

                def line(i, tzinfo=None):
                    return [
                        i,
                        test_date(),
                        test_time(tzinfo=tzinfo),
                        test_time(tzinfo=tzinfo),
                        test_datetime(tzinfo=tzinfo),
                        test_datetime(tzinfo=tzinfo),
                    ]

                test_data = [
                    line(i, tzinfo) for i, tzinfo in enumerate(
                        [None, 2, 3, "Europe/Rome", "Zulu"])
                ]

                connection.execute(table.insert(), [
                    dict(zip(table.c.keys(), test_value))
                    for test_value in test_data
                ])

                #after inserting the data we make sure that what is coming out from sqlalchemy got the correct type
                #and that the str repr makes sense, this is just a shortcut to avoid writing a lot of code

                for values, tests in zip(
                        connection.execute(sa.select([table])), test_data):

                    for column, value, test in zip(table.c.values(), values,
                                                   tests):

                        if isinstance(test,
                                      (datetime.datetime, datetime.time)):

                            self.assertEqual(
                                bool(value.tzinfo),
                                bool(column.type.timezone),
                                msg='%s field is timezone %s, value %s is not'
                                % (column.type, column.type.timezone
                                   and 'aware' or 'naive', value))

                        if isinstance(test,
                                      (datetime.time, datetime.datetime)):

                            self.assertEqual(
                                bool(value.tzinfo), bool(column.type.timezone))

                        if isinstance(test, datetime.time):
                            if column.type.timezone:

                                if not test.tzinfo:

                                    #the field was naive, so if we strip tzinfo we should get back the naive date

                                    self.assertEqual(
                                        new_time(value, tzinfo=None), test)

                            else:

                                self.assertEqual(value,
                                                 new_time(value, tzinfo=None))

                        elif isinstance(test, datetime.datetime):

                            if column.type.timezone:

                                if not test.tzinfo:

                                    #the field was naive, so if we strip tzinfo we should get back the naive date

                                    self.assertEqual(
                                        new_datetime(
                                            value, value, tzinfo=None), test)
                                else:

                                    self.assertEqual(
                                        pytz.utc.normalize(test),
                                        pytz.utc.normalize(value))
                            else:

                                self.assertEqual(
                                    value,
                                    new_datetime(value, value, tzinfo=None))

                        elif isinstance(test, datetime.date):

                            self.assertEqual(test, value)
