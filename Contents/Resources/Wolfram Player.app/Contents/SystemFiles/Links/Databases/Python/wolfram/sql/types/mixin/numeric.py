# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import decimal

from wolfram.sql import sqlalchemy as sa
from wolfram.sql.types.mixin.base import WolframType, construct_base_type
from wolframclient.utils import six

#NUMERIC FIELDS


class Integer(WolframType):
    """Platform-independent GUID type.
    Uses Postgresql's UUID type, otherwise uses
    BigInteger, storing as integer the uuid value.
    """

    impl = construct_base_type(sa.types.Integer)

    def process_result_value(self, value, dialect):

        if self.is_null(value):
            return None

        if isinstance(value, bool):
            #in python True is an instance of int
            return self.invalid_value(value)

        if isinstance(value, six.integer_types):
            return value

        if isinstance(value, (float, decimal.Decimal, six.string_types)):
            try:
                return int(value)
            except ValueError:
                pass

        return self.invalid_value(value)


class Real(WolframType):

    impl = construct_base_type(sa.types.REAL)

    def load_dialect_impl(self, dialect):

        if dialect.name == 'mssql':
            return sa.databases.mssql.FLOAT(53)

        return super(Real, self).load_dialect_impl(dialect)

    def process_result_value(self, value, dialect):

        if self.is_null(value):
            return None

        if isinstance(value, float):
            return value

        if isinstance(value, bool):
            #in python True is an instance of int
            return self.invalid_value(value)

        if isinstance(value, (int, decimal.Decimal, six.string_types)):
            try:
                return float(value)
            except ValueError:
                pass

        return self.invalid_value(value)


class Decimal(WolframType):

    #Type for decimal with arbitrary precision

    impl = construct_base_type(sa.types.Numeric)

    def process_result_value(self, value, dialect):

        if self.is_null(value):
            return None

        if isinstance(value, decimal.Decimal):
            return value

        if isinstance(value, bool):
            #in python True is an instance of int
            return self.invalid_value(value)

        if isinstance(value, (float, int, six.string_types)):
            try:
                return decimal.Decimal(value)
            except (ValueError, decimal.InvalidOperation):
                pass

        return self.invalid_value(value)
