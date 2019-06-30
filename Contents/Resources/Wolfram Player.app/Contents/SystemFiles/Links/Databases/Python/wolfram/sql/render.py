# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import datetime

from wolframclient.utils import six


def render_query(statement, bind=None):
    """

    Borrewed from: https://gist.github.com/gsakkis/4572159

    Generate an SQL expression string with bound parameters rendered inline
    for the given SQLAlchemy statement.
    WARNING: This method of escaping is insecure, incomplete, and for debugging
    purposes only. Executing SQL statements with inline-rendered user values is
    extremely insecure.
    Based on http://stackoverflow.com/questions/5631078/sqlalchemy-print-the-actual-query
    """

    if isinstance(statement, six.string_types):
        return statement

    if bind is None:
        bind = statement.bind

    class LiteralCompiler(bind.dialect.statement_compiler):
        def visit_bindparam(self,
                            bindparam,
                            within_columns_clause=False,
                            literal_binds=False,
                            **kwargs):
            return self.render_literal_value(bindparam.value, bindparam.type)

        def render_literal_value(self, value, type_):
            if isinstance(value, (datetime.date,
                                    datetime.datetime, datetime.time)):
                return "'%s'" % value
            return super(LiteralCompiler, self).render_literal_value(
                value, type_)

    return LiteralCompiler(bind.dialect, statement).process(statement)
