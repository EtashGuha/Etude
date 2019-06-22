# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from sqlalchemy.dialects.postgresql.zxjdbc import PGDialect_zxjdbc as Dialect
from sqlalchemy.dialects.postgresql.zxjdbc import \
    PGExecutionContext_zxjdbc as Context

#for some reason in jython _is_server_side is not an attribute of the context, we need to hardcode this to False if we want postgres to work
Context._is_server_side = False

dialect = Dialect
