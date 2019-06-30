# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from sqlalchemy.dialects.oracle.base import OracleDialect, TIMESTAMP, RAW, CHAR

OracleDialect.ischema_names.update({
    'TIMESTAMP WITH LOCAL TIME ZONE': TIMESTAMP,
    'LONG RAW': RAW,
    'ROWID': CHAR,
    'UROWID': CHAR,
    'NCHAR': CHAR,
    'BLOB': RAW,
    'BFILE': RAW,
})

class OracleMixin(object):
    def __init__(self, exclude_tablespaces=('SYSAUX', ), **opts):
        super(OracleMixin, self).__init__(
            exclude_tablespaces=exclude_tablespaces, **opts)

    def on_connect(self):
        func = super(OracleMixin, self).on_connect()

        def inner(conn):
            if func:
                func(conn)
            conn.cursor().execute("alter session set time_zone='UTC'")

        return inner
