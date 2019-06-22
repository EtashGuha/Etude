# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from wolfram.execution.messages import messages
from wolfram.sql import sqlalchemy as sa
from wolfram.sql.types import database_types
from wolframclient.language.expression import WLFunction
from wolframclient.utils import six
from wolframclient.utils.functional import is_iterable, iterate


def _collections_for_dialect(dialect=None):

    if isinstance(dialect, six.string_types):
        #if the dialect is a string we are just doing a getattr of from that module, there are defined attrs including sqlite, mysql, etc...

        yield getattr(database_types, dialect)

    elif dialect:
        #those dialects can be a class of default.DefaultDialect and in that case they are defining a name attribute that the implemented database (mysql, sqlite, ...)

        yield getattr(database_types, dialect.name)

    #defined SqlAlchemy high-level types dialect independent (normally camel-cased)
    yield database_types.wolfram


def get_type_info(type_, dialect=None):

    if getattr(type_, 'type_info', None):
        return type_.type_info

    if isinstance(type_, sa.types.TypeEngine):
        type_ = type_.__class__.__name__

    for collection in _collections_for_dialect(dialect):
        try:
            return collection[type_]
        except KeyError:
            pass

    if type_ == 'Unsupported':
        #null type is returned when SQLAlchemy is not able to infer the type
        return database_types.wolfram.Unsupported

    return database_types.wolfram.Unsupported


def _normalize_type_spec(type_name, default=None):
    if isinstance(type_name, six.string_types):
        return (type_name, )
    elif isinstance(type_name, WLFunction):
        return iterate(*type_name.args)
    elif is_iterable(type_name):
        if len(type_name) == 0:
            return (default, )
        return type_name
    elif type_name is None:
        return (default, )

    raise TypeError('Type name should be a string and not %s' % type_name)


def _to_database_type(dialect, database_type, *args, **kw):
    if database_type:

        info = get_type_info(database_type, dialect=dialect)

        if not info:
            raise messages.type_is_invalid.as_exception(database_type)

        return info(*args, **kw)


def to_database_type(dialect, database_type, default='String'):

    if isinstance(database_type, sa.types.TypeEngine):
        return database_type

    return _to_database_type(
        dialect, *_normalize_type_spec(database_type, default=default))
