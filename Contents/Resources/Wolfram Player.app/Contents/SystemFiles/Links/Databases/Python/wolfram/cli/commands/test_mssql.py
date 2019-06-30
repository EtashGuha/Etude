# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from wolfram.cli.dispatch import SimpleCommand
from wolfram.sql import sqlalchemy as sa
from wolfram.utils.normalize import serialize_metadata


class Command(SimpleCommand):
    def handle(self, **opts):

        engine = sa.create_engine(
            "mssql://WolframUser:!990oZyWfLBvq2rEz5SFcSKiGFy0=@mssql-test1/")
        metadata = sa.MetaData(bind=engine)
        metadata.reflect(only=['person'])

        print(serialize_metadata(metadata))
