# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals
from wolframclient.utils import six
from wolframclient.utils.decorators import to_tuple
from wolframclient.utils.functional import first
import re




@to_tuple
def to_version(version, number_re = re.compile(r'\d+'), mariadb_pattern = re.compile(r'(.*)-mariadb$', re.IGNORECASE)):
    if isinstance(version, six.binary_type):
        version = version.decode()

    for v in number_re.findall(version):
        yield int(v)

    if mariadb_pattern.match(version, re.IGNORECASE):
        yield 'MariaDB'

class MysqlMixin(object):
    def on_connect(self):
        func = super(MysqlMixin, self).on_connect()

        def inner(conn):
            if func:
                func(conn)
            conn.cursor().execute('SET @@session.time_zone = "+00:00"')

        return inner

    #we need to patch _get_server_version_info in python3
    #sqlalchemy might fetch something like 3.12a.2 and then comparing with < (2, 3, 0).
    #this is raising an error because '12a' cannot be compared with 3 

    #the following implementation would return (3, 12, 2) for 3.12a.2 instead of (3, '12a', 2)

    def _get_server_version_info(self, connection):
        # get database server version info explicitly over the wire
        # to avoid proxy servers like MaxScale getting in the
        # way with their own values, see #4205

        with connection.connection.cursor() as cursor:
            cursor.execute("SELECT VERSION()")
            return to_version(first(cursor.fetchone()))