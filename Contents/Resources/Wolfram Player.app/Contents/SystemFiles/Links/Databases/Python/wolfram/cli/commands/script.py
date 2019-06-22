# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import datetime

from wolfram.sql import sqlalchemy as sa
from wolfram.sql.compilable import Compilable
from wolfram.sql.render import render_query
from wolframclient.cli.utils import SimpleCommand


class Command(SimpleCommand):
    def handle(self):

        #SHOULD NOT BE MERGED WITH MASTER

        qs = sa.select([Compilable(datetime.datetime.now())])

        with sa.create_engine('sqlite://').connect() as c:

            print(render_query(qs, c))

            for row in c.execute(qs):
                print(row[0], row[0].__class__)
