# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from sqlalchemy import cast as _cast

from wolfram.sql.types.utils import to_database_type


def cast(expr, database_type, dialect=None):
    return _cast(
        expr, to_database_type(dialect=dialect, database_type=database_type))
