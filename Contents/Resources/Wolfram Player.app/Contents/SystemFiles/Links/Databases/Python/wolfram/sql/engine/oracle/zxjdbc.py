# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from sqlalchemy.dialects.oracle.zxjdbc import OracleDialect_zxjdbc

from wolfram.sql.engine.oracle.mixin import OracleDialect


class OracleDialect(OracleDialect, OracleDialect_zxjdbc):
    pass


dialect = OracleDialect
