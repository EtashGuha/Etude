# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import os
from collections import defaultdict

from sqlalchemy import create_engine as _create_engine
from sqlalchemy import pool, util
from sqlalchemy.dialects import registry
from sqlalchemy.engine.url import URL, make_url
from sqlalchemy.sql.base import DialectKWArgs, _DialectArgView

from wolfram.execution.messages import messages
from wolfram.utils.normalize import serialize_url
from wolframclient.language import wl
from wolframclient.serializers.serializable import WLSerializable
from wolframclient.utils import six
from wolframclient.utils.decorators import to_dict
from wolframclient.utils.encoding import force_text

SPELLINGS = {
    'postgresql': ('postgres', 'postgresql', 'postgressql'),
    'sqlite': ('sqlite', 'sqllite'),
    'mssql': ('mssql', 'msql', 'microsoftsql', 'microsoft', 'ms'),
    'oracle': ('oracle', 'oraclesql', )
}

JYTHON_DRIVERS = {
    'postgresql': "wolfram.sql.engine.postgresql.zxjdbc",
    "sqlite": "wolfram.sql.engine.sqlite.zxjdbc",
    "mysql": "wolfram.sql.engine.mysql.zxjdbc",
    "oracle": "wolfram.sql.engine.oracle.zxjdbc",
    "mssql": "wolfram.sql.engine.mssql.zxjdbc",
}

PYTHON_DRIVERS = {
    'postgresql': "wolfram.sql.engine.postgresql.psycopg2",
    "sqlite": "wolfram.sql.engine.sqlite.pysqlite",
    "mysql": "wolfram.sql.engine.mysql.pymysql",
    "oracle": "wolfram.sql.engine.oracle.cx_oracle",
    "mssql": "wolfram.sql.engine.mssql.pymssql",
}


def drivers(drivers, spellings=SPELLINGS):
    for name, default in drivers.items():
        for spelling in (spellings.get(name) or (name, )):
            yield spelling, default or template % name


def available_dialects():
    #this is adding working zxjdbc even if python is CPython
    for key, value in drivers(JYTHON_DRIVERS):
        yield '%s.jdbc' % key, value
        if six.JYTHON:
            #if we are using jython then the default dialect is zxjdbc
            yield key, value

    if not six.JYTHON:
        #if we are using python there we want to register some overrides
        for key, value in drivers(PYTHON_DRIVERS):
            yield key, value


for name, path in available_dialects():
    registry.register(name, path, "dialect")

if six.JYTHON:

    #this is a patch for JYthon, the jython interpreter is not handling very well the __getattr__ method of a custom dict class.
    #with this custom method instead of returning a custom dict we are statically building a normal dict in order to avid this bug.

    @util.memoized_property
    def to_safe_kwargs(self):
        #we are using immutable dict because this cannot be changed anymore
        return util.immutabledict(_DialectArgView(self))

    DialectKWArgs.dialect_kwargs = to_safe_kwargs

ENGINE_KEY = 'e'
CONNEC_KEY = 'c'


class EngineManager(object):

    pool_class = pool.SingletonThreadPool

    def __init__(self):
        self._connections = defaultdict(dict)

    #connection iterable and close
    def __iter__(self):
        for connection in self._connections.keys():
            if self.is_connected(connection):
                yield connection

    def __len__(self):
        return sum(1 for c in self)

    def __repr__(self):
        return '<%s %s>' % (self.__class__.__name__, list(self))

    def create_engine(self, connection):
        try:
            return self._connections[connection][ENGINE_KEY]
        except KeyError:

            self.check_connection(connection)
            self._connections[connection][ENGINE_KEY] = _create_engine(
                connection.url, poolclass=self.pool_class)
            return self._connections[connection][ENGINE_KEY]

    def check_connection(self, connection):
        if connection.url.drivername == 'sqlite' \
            and connection.url.database \
            and not os.path.exists(connection.url.database):

            raise messages.sqlite_missing_file.as_exception(
                connection.url.database)

    def begin(self, connection):
        return self.connect(connection).begin()

    def is_connected(self, connection):
        try:
            return bool(not self._connections[connection][CONNEC_KEY].closed)
        except KeyError:
            return False

    def connect(self, connection):
        if not self.is_connected(connection):
            self._connections[connection][CONNEC_KEY] = self.create_engine(
                connection).connect()

        return self._connections[connection][CONNEC_KEY]

    def disconnect(self, connection):
        try:
            self._connections[connection][CONNEC_KEY].close()
        except KeyError:
            pass

        try:
            self._connections[connection][ENGINE_KEY].dispose()
        except KeyError:
            pass

        try:
            del self._connections[connection]
        except KeyError:
            pass

    def execute(self, connection, *args, **kw):
        return self.connect(connection).execute(*args, **kw)

    def disconnect_all(self):
        for connection in self._connections.keys():
            self.disconnect(connection)

        self._connections = defaultdict(dict)

    def __enter__(self):
        self.disconnect_all()
        return self

    def __exit__(self, type, value, tb):
        self.disconnect_all()


engines = EngineManager()


class Connection(WLSerializable):
    def __init__(self, url, id=None, manager=engines):
        self.url = self.validate(url)
        self.manager = manager
        self.id = id

    def validate(self, url):
        if url.drivername:
            url.drivername = url.drivername.lower()

        return url

    def to_wl(self, *args, **opts):
        return wl.DatabaseReference(self._connection_properties())

    @to_dict
    def _connection_properties(self):
        yield "ID", force_text(self.id)
        for key, value in serialize_url(self.url).items():
            yield key, value

    def __repr__(self):
        return '<%s %s>' % (self.__class__.__name__, self.id)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Connection) and self.id == other.id

    def get_id(self):
        if not self._id:
            self._id = 'id[%s]' % self.url
        return self._id

    def set_id(self, value):
        self._id = value

    id = property(get_id, set_id)

    def create_engine(self, *args, **kw):
        return self.manager.create_engine(self, *args, **kw)

    def begin(self, *args, **kw):
        return self.manager.begin(self, *args, **kw)

    def connect(self, *args, **kw):
        return self.manager.connect(self, *args, **kw)

    def disconnect(self, *args, **kw):
        return self.manager.disconnect(self, *args, **kw)

    def is_connected(self, *args, **kw):
        return self.manager.is_connected(self, *args, **kw)

    def execute(self, *args, **kw):
        return self.manager.execute(self, *args, **kw)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()


def create_connection(url, **opts):
    if isinstance(url, six.string_types):
        #need to convert to URL
        return Connection(url=make_url(url), **opts)

    if isinstance(url, dict):
        return Connection(
            id=url.get('ID', None),
            url=URL(
                (url.get("Backend", None) or 'sqlite').lower(),
                username=url.get('Username', None),
                password=url.get('Password', None),
                host=url.get('Host', None),
                port=url.get('Port', None),
                database=url.get('Name', None),
                query=url.get('Options', None),
            ),
            **opts)
    if isinstance(url, Connection):
        return url
    if isinstance(url, URL):
        return Connection(url=url, **opts)

    raise NotImplementedError('Cannot create an engine from %s' % url)


def create_engine(url, **opts):
    return create_connection(url, **opts).create_engine()
