# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from sqlalchemy.dialects.mssql.pymssql import MSDialect_pymssql

from wolfram.sql.engine.mssql.mixin import MSMixin


class MSDialect(MSMixin, MSDialect_pymssql):
    pass


dialect = MSDialect
