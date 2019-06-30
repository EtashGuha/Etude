# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from sqlalchemy.dialects.mysql.zxjdbc import MySQLDialect_zxjdbc

from wolfram.sql.engine.mysql.mixin import MysqlMixin


class MysqlDialect(MysqlMixin, MySQLDialect_zxjdbc):
    pass


dialect = MysqlDialect
