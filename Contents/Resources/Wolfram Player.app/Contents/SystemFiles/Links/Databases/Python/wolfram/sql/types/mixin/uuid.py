# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import uuid

from wolfram.sql import sqlalchemy as sa
from wolfram.sql.types.mixin.base import WolframType, construct_base_type
from wolframclient.utils import six
from wolframclient.utils.encoding import force_text

if six.JYTHON:
    import java.util.UUID


class UUID(WolframType):

    impl = construct_base_type(sa.types.String)
    """Platform-independent GUID type.
    Uses Postgresql's UUID type, otherwise uses
    BigInteger, storing as integer the uuid value.
    """

    def load_dialect_impl(self, dialect):

        if dialect.name == 'postgresql':
            return dialect.type_descriptor(sa.databases.postgresql.UUID())

        if dialect.name == 'mysql':
            return dialect.type_descriptor(sa.databases.mysql.BINARY(16))

        return dialect.type_descriptor(sa.types.String(32))

    def to_uuid(self, value, dialect):

        if self.is_null(value):
            return None

        if six.JYTHON and isinstance(value, java.util.UUID):
            return uuid.UUID(hex=force_text(value))

        if isinstance(value, uuid.UUID):
            return value

        if isinstance(value, six.string_types):

            if len(value) in (32, 36):
                return uuid.UUID(hex=value)
            return uuid.UUID(bytes=value)

        if isinstance(value, six.integer_types):
            return uuid.UUID(int=value)

        return self.invalid_value(value)

    def process_bind_param(self, value, dialect):

        value = self.to_uuid(value, dialect) or uuid.uuid4()

        if dialect.name == 'mysql':
            return value.bytes

        if dialect.name == 'postgresql' and six.JYTHON:
            return java.util.UUID.fromString(force_text(value))

        return value.hex

    def process_result_value(self, value, dialect):
        return self.to_uuid(value, dialect)
