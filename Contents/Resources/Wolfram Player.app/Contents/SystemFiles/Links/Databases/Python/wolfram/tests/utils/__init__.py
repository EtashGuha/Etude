# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from wolframclient.utils.importutils import API

TESTS = API(
    serialization='wolfram.tests.serialization.SerializationCase',
    types='wolfram.tests.types.TypeCase',
    internals='wolfram.tests.internals.InternalsCase',
    connection='wolfram.tests.connection.ConnectionCase',
    inspection='wolfram.tests.inspection.InspectionCase',
    timezone='wolfram.tests.timezone.TimezoneCase',
)


def available_test_suites():
    return tuple(TESTS.keys())


def run_test_suite(*args):
    import unittest

    suite = unittest.TestSuite()

    for val in (args or TESTS):
        for func in TESTS[val].discover_tests():
            suite.addTest(TESTS[val](func))

    runner = unittest.TextTestRunner()
    runner.run(suite)
