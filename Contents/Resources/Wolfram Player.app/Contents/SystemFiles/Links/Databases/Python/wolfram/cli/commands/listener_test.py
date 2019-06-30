# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from wolfram.cli.dispatch import SimpleCommand
from wolfram.sql import sqlalchemy as sa
from wolframclient.utils.decorators import to_tuple
from wolframclient.utils.functional import first, iterate, last
from wolframclient.serializers import export
import platform

def same_import(imports):
    for sub in iterate(imports):
        try:
            return __import__(sub)
        except ImportError:
            pass
    return None

def dependencies():
    yield 'wolframclient',
    yield 'sqlalchemy',
    yield 'pytz',
    yield 'pymssql',
    yield 'psycopg2',
    yield 'pymysql',
    yield ('pysqlite2', 'sqlite3', 'sqlite')   #sqlalchemy is testing those 3 in this order


def sqlalchemy_urls():
    yield "sqlite://"
    yield "postgres://localhost"
    yield "mysql://localhost"
    yield "mssql://localhost"


class Command(SimpleCommand):

    @to_tuple
    def test_all(self):
        for module in dependencies():
            yield first(iterate(module)), same_import(module)

    def handle(self, **opts):

        self.print('TESTING WOLFRAMCLIENT:')

        assert export([1,2,3], target_format='wxf') == b'8:f\x03s\x04ListC\x01C\x02C\x03'
        
        self.print('TESTING DEPENDENCIES:')

        checks = self.test_all()

        for module, check in checks:
            self.print(' -', module, check and "OK" or "FAILED")

        if not all(map(last, checks)):
            raise ImportError('Cannot import the followings: %s' % ", ".join(
                module for module, check in checks if not check))

        self.print('TESTING pytz')

        import pytz
        import datetime

        formatted = pytz.utc.localize(
            datetime.datetime(2002, 10, 27, 6, 0, 0)).isoformat()

        assert formatted == "2002-10-27T06:00:00+00:00", "Cannot format pytz date properly"

        self.print('TESTING pymysql')

        from pymysql import STRING as PYMYSQL_STRING

        print(PYMYSQL_STRING)  #this is checking the api is there and working

        self.print('TESTING psycopg2')

        from psycopg2 import STRING

        print(STRING)

        self.print('TESTING mssql')

        import pymssql

        self.print(pymssql.get_dbversion())

        self.print('TESTING sqlite')

        #sqlite is the only one that can be tested in memory, doing that

        from sqlalchemy import Column, Table, MetaData, Integer, String, select

        from wolfram.sql.render import render_query

        with sa.create_engine("sqlite://").connect() as connection:
            for a, b in connection.execute("SELECT 1, 2").fetchall():
                assert a == 1 and b == 2

        table = Table(
            "table",
            MetaData(),
            Column("id", Integer(), primary_key=True),
            Column("name", String()),
        )

        for url in sqlalchemy_urls():

            self.print('TESTING %s' % url)

            print(
                render_query(
                    select([table.c.id, table.c.name]),
                    bind=sa.create_engine(url)))
