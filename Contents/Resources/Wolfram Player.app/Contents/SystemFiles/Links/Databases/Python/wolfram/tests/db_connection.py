# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import os
import tempfile

from wolfram.execution.evaluator import WolframLanguageEvaluator
from wolfram.sql import sqlalchemy as sa
from wolfram.tests.utils.base import TestCase as BaseTestCase
from wolframclient.language import wl


class TestCase(BaseTestCase):

    available_engines = [
        'sqlite://',
        {
            "Backend": 'sqlite',
            'Name': os.path.join(tempfile.gettempdir(), 'temp-db1.sqlite')
        },
        {
            "Backend": 'sqlite',
            'Name': os.path.join(tempfile.gettempdir(), 'temp-db2.sqlite')
        }
    ]

    def test_unique_connection(self):

        engine = 'sqlite://'

        with sa.engines as engines:

            self.assertEqual(len(engines), 0)

            c1 = sa.create_connection(engine, manager=engines)

            self.assertEqual(len(engines), 0)

            self.assertEqual(c1.is_connected(), False)

            c1.connect()

            self.assertEqual(c1.is_connected(), True)

            self.assertEqual(len(engines), 1)

            c2 = sa.create_connection(engine, manager=engines)

            self.assertEqual(len(engines), 1)

            self.assertEqual(c2.is_connected(), True)

            c2.connect()

            self.assertEqual(c1.is_connected(), True)
            self.assertEqual(c2.is_connected(), True)

            self.assertEqual(len(engines), 1)

            c1.disconnect()

            self.assertEqual(len(engines), 0)

            self.assertEqual(c1.is_connected(), False)
            self.assertEqual(c2.is_connected(), False)

    def test_multiple_engines(self):

        with sa.engines as engines:

            self.assertEqual(len(engines), 0)

            connections = [
                sa.create_connection(engine, manager=engines)
                for engine in self.available_engines
            ]

            self.assertEqual(len(engines), 0)

            for i, connection in enumerate(connections):

                self.assertEqual(len(engines), i)

                connection.connect()

                self.assertEqual(len(engines), i + 1)

            self.assertEqual(len(engines), len(connections))

            for i, connection in enumerate(connections):

                self.assertEqual(len(engines), len(connections) - i)

                connection.disconnect()

                self.assertEqual(len(engines), len(connections) - i - 1)

            self.assertEqual(len(engines), 0)

    def evaluate_wl_command(self, command, engine=None):
        with WolframLanguageEvaluator() as evaluator:
            return evaluator.evaluate(
                wl.WithEnvironment(command, {"connection": engine}))

    def test_wl_engines(self):

        with sa.engines as engines:

            for engine in self.available_engines:
                self.assertEqual(
                    self.evaluate_wl_command(wl.DatabaseConnected(), engine),
                    False)

            self.assertEqual(len(engines), 0)

            for i, engine in enumerate(self.available_engines):
                self.evaluate_wl_command(wl.DatabaseConnect(), engine)
                self.assertEqual(len(engines), i + 1)

            self.assertEqual(
                len(engines),
                len(self.available_engines),
            )

            for engine in self.available_engines:
                self.assertEqual(
                    self.evaluate_wl_command(wl.DatabaseConnected(), engine),
                    True)

            for i, engine in enumerate(self.available_engines):
                self.evaluate_wl_command(wl.DatabaseDisconnect(), engine)
                self.assertEqual(
                    len(engines),
                    len(self.available_engines) - i - 1)

            self.assertEqual(len(engines), 0)
