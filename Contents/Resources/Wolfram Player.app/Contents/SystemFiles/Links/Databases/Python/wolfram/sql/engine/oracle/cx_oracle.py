# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from sqlalchemy.dialects.oracle.cx_oracle import OracleDialect_cx_oracle

from wolfram.sql.engine.oracle.mixin import OracleMixin


class OracleDialect(OracleMixin, OracleDialect_cx_oracle):
    pass


dialect = OracleDialect
