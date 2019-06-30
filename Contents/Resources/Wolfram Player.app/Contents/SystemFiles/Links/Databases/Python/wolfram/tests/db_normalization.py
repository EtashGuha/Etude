# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from wolfram.sql import sqlalchemy as sa
from wolfram.tests.utils.base import TestCase as BaseTestCase
from wolfram.utils.normalize import dumps
from wolframclient.serializers import available_formats, export


class TestCase(BaseTestCase):
    def test_normalization(self):

        metadata = sa.MetaData(bind=sa.create_engine('sqlite:///:memory:'))

        user = sa.Table(
            'user', metadata,
            sa.Column('user_id', sa.types.Integer(), primary_key=True),
            sa.Column('user_name', sa.types.String(16), nullable=False),
            sa.Column('email_address', sa.types.String(60)),
            sa.Column('password', sa.types.String(20), nullable=False))

        for f in available_formats:

            with self.assertRaises(Exception) as context:
                export(metadata, target_format=f)

            export(metadata, target_format=f)
            dumps(metadata, target_format=f)
