# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from wolfram.tests.utils.base import TestCase as BaseTestCase
from wolfram.utils.dates import to_timezone
from wolframclient.serializers import export


class TestCase(BaseTestCase):
    def test_tz_conversion(self):

        #The value 'SYSTEM' indicates that the time zone should be the same as the system time zone.
        #The value can be given as a string indicating an offset from UTC, such as '+10:00' or '-6:00'.
        #The value can be given as a named time zone, such as 'Europe/Helsinki', 'US/Eastern', or 'MET'. Named time zones can be used only if the time zone information tables in the mysql database have been created and populated.

        for spec, res in (('+10:00', b'10.0'), ('-10:00', b'-10.0'),
                          ('-10', b'-10.0'), ('10',
                                              b'10.0'), ('Europe/Helsinki',
                                                         b'"Europe/Helsinki"'),
                          ('US/Eastern', b'"US/Eastern"'), ('MET', b'1.0')):

            exported = export(to_timezone(spec, allow_string_offset=True))

            print(spec, exported, res)

            self.assertEqual(res, exported)
