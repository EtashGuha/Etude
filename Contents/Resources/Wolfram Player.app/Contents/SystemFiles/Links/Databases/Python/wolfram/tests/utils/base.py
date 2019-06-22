# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from contextlib import contextmanager

from wolfram.sql import sqlalchemy as sa
from wolframclient.utils.functional import iterate
from wolframclient.utils.tests import TestCase as BaseTestCase


class TestCase(BaseTestCase):
    def default_engines(self):

        for url in (
                'sqlite:///:memory:',
                #'mysql://root@localhost/wolfram_test',
                #'postgres://rdv@localhost/wolfram_test',
        ):

            yield url

    @contextmanager
    def _create_test_database(self, init, engine, delete=False):

        metadata = sa.MetaData(bind=sa.create_engine(engine))

        init(metadata)
        metadata.drop_all()
        metadata.create_all()

        yield metadata

        if delete:
            metadata.drop_all()

        metadata.bind.dispose()

    def create_test_databases(self, init, engine=None, **opts):

        for eng in iterate(engine or self.default_engines()):
            with self._create_test_database(init, eng) as metadata:
                yield metadata
