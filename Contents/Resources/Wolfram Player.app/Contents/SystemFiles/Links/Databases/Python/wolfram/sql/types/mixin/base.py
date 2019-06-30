# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from wolfram.sql import sqlalchemy as sa
from wolfram.sql.types import database_types as types
from wolframclient.language import wl
from wolframclient.serializers.serializable import WLSerializable


def dialect_impl(self, dialect):

    #this is using sqlalchemy implementation, dialect.colspecs has a series of classes that are mapped to internal classes
    #we as convention are using the very same class name, so we can just use that to recursevly load our internal class, if defined

    impltype = info = None

    for t in self.__class__.__mro__[0:-1]:
        try:
            impltype = dialect.colspecs[t]
            break
        except KeyError:
            pass

    if impltype:
        try:
            info = types[dialect.name][impltype.__name__]
        except KeyError:
            pass

    if info:
        return self.adapt(info.db_type)

    return super(self.__class__, self).dialect_impl(dialect)


def construct_base_type(base):
    return type(base.__class__.__name__, (base, ),
                {'dialect_impl': dialect_impl})


class InvalidResult(WLSerializable):
    def __init__(self, value, type):
        self.value = value
        self.type = type

    def __bool__(self):
        return False

    def __repr__(self):
        return '<InvalidResult %s for type %s>' % (self.value, self.type)

    def to_wl(self, *args, **opts):
        return wl.Failure("InvalidResult", {
            "Type": self.type,
            "Value": self.value
        })


class WolframType(sa.types.TypeDecorator):
    """

    FROM SQLALCHEMY SOURCE CODE

    Allows the creation of types which add additional functionality
    to an existing type.

    This method is preferred to direct subclassing of SQLAlchemy's
    built-in types as it ensures that all required functionality of
    the underlying type is kept in place.

    Typical usage::

    import sqlalchemy.types as types

    class MyType(types.TypeDecorator):
        '''Prefixes Unicode values with "PREFIX:" on the way in and
        strips it off on the way out.
        '''

        impl = types.Unicode

        def process_bind_param(self, value, dialect):
            return "PREFIX:" + value

        def process_result_value(self, value, dialect):
            return value[7:]

        def copy(self, **kw):
            return MyType(self.impl.length)

    The class-level "impl" attribute is required, and can reference any
    TypeEngine class.  Alternatively, the load_dialect_impl() method
    can be used to provide different type classes based on the dialect
    given; in this case, the "impl" variable can reference
    ``TypeEngine`` as a placeholder.

    """

    is_wl_type = True
    type_info = None

    def is_null(self, value):
        return value is None or value is ''

    def invalid_value(self, value):
        return InvalidResult(value, self)

    def process_bind_param(self, value, dialect):
        #Wolfram side type system will ensure this is already compiled, no need for type checking
        if isinstance(value, sa.Compilable):
            return value.instance
        return value
