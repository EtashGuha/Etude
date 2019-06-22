# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import datetime

from wolfram.sql.types import database_types as types
from wolfram.tests.utils.base import TestCase as BaseTestCase
from wolfram.tests.utils.dates import test_date, test_datetime, test_time
from wolfram.utils.dates import new_datetime, new_time, pytz
from wolframclient.utils.functional import first, last


class TestCase(BaseTestCase):
    def compare_date(self,
                     date,
                     string_version='',
                     fmt='%Y-%m-%d %H:%M:%S %Z%z'):
        return self.assertEqual(date.strftime(fmt).strip(), string_version)

    def test_timezone_conversion(self):

        self.compare_date(test_datetime(tzinfo=None), '2000-01-01 11:15:20')

        self.compare_date(test_datetime(tzinfo=1), '2000-01-01 11:15:20 +0100')

        self.compare_date(
            test_datetime(tzinfo='Europe/Amsterdam'),
            '2000-01-01 11:15:20 CET+0100')

        with self.assertRaises(pytz.UnknownTimeZoneError):
            self.compare_date(test_datetime(tzinfo='some/nonsense'))

    def test_python_conversion(self):

        values = lambda data: map(last, sorted(data.items(), key=first))

        all_types = [
            types.wolfram.Date(),
            types.wolfram.Time(False),
            types.wolfram.Time(True),
            types.wolfram.DateTime(False),
            types.wolfram.DateTime(True),
        ]

        def line(tzinfo=None):
            return [
                test_date(),
                test_time(tzinfo=tzinfo),
                test_time(tzinfo=tzinfo),
                test_datetime(tzinfo=tzinfo),
                test_datetime(tzinfo=tzinfo),
            ]

        test_data = map(line, [None, 2, 3, "Europe/Rome", "Zulu"])

        for data in test_data:
            for type, test in zip(all_types, data):

                value = type.process_result_value(test, dialect=None)

                if isinstance(test, (datetime.datetime, datetime.time)):

                    if type.timezone:

                        self.assertEqual(value, test)

                    else:

                        self.assertEqual(bool(value.tzinfo), False)

                        if isinstance(test, datetime.datetime):
                            self.assertEqual(value, new_datetime(test, test))

                        elif isinstance(test, datetime.time):
                            self.assertEqual(value, new_time(test))

                elif isinstance(test, datetime.date):
                    self.assertEqual(test, value)
