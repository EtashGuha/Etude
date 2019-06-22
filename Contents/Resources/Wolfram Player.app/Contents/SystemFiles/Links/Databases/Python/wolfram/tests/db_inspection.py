# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import datetime

from wolfram.execution.evaluator import WolframLanguageEvaluator
from wolfram.sql import sqlalchemy as sa
from wolfram.tests.utils.base import TestCase as BaseTestCase
from wolfram.utils.normalize import dumps
from wolframclient.language import wl


class TestCase(BaseTestCase):
    def create_db_for_inspection(self, metadata):
        sa.Table(
            'user',
            metadata,
            sa.Column('user_id', sa.types.Integer(), primary_key=True),
            sa.Column('date', sa.types.Date(), nullable=False),
        )

    def test_inspection(self):

        for metadata in self.create_test_databases(
                self.create_db_for_inspection):

            with WolframLanguageEvaluator() as evaluator:

                result = evaluator.evaluate(
                    wl.WithEnvironment(
                        wl.DatabaseInsert({
                            "user": [{
                                "user_id": 1,
                                "date": datetime.datetime.now()
                            }]
                        }), {
                            'path': None,
                            "connection": metadata.bind.url,
                            "metadata": wl.All
                        }))

                print(dumps(result, indent=4))

                result = evaluator.evaluate(
                    wl.WithEnvironment(
                        wl.DatabaseDump(), {
                            'path': None,
                            "connection": metadata.bind.url,
                            "metadata": wl.All
                        }))

                print(dumps(result, indent=4))
