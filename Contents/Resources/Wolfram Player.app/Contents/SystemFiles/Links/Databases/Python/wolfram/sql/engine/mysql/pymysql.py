# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from sqlalchemy.dialects.mysql.pymysql import MySQLDialect_pymysql

from wolfram.sql.engine.mysql.mixin import MysqlMixin


class MysqlDialect(MysqlMixin, MySQLDialect_pymysql):
    pass


dialect = MysqlDialect
