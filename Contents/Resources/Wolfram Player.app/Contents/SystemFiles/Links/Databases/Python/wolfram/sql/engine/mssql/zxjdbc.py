# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from sqlalchemy.dialects.mssql.zxjdbc import MSDialect_zxjdbc

from wolfram.sql.engine.mssql.mixin import MSDialect


class MSDialect(MSDialect, MSDialect_zxjdbc):
    pass


dialect = MSDialect
