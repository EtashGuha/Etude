# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import datetime

from wolfram.sql import sqlalchemy as sa
from wolfram.sql.types.mixin.base import WolframType, construct_base_type
from wolfram.sql.types.mixin.numeric import Real
from wolfram.utils.dates import new_datetime, pytz

EPOCH = new_datetime(datetime.date(year=1970, month=1, day=1), tzinfo=0)


def to_utc(value):
    if value.__class__ == datetime.date:
        value = new_datetime(value)
    if value.tzinfo:
        return pytz.utc.normalize(value)
    return pytz.utc.localize(value)


class NativeDateTime(WolframType):

    impl = construct_base_type(sa.types.DateTime)

class NativeTime(WolframType):

    impl = construct_base_type(sa.types.Time)


class NativeDate(WolframType):

    impl = construct_base_type(sa.types.Date)


class NativeTimeQuantity(WolframType):

    impl = construct_base_type(sa.types.Interval)


class Date(WolframType):

    impl = NativeDate.impl

    def load_dialect_impl(self, dialect):

        if dialect.name == 'sqlite':
            return Real()

        return NativeDate()

    def compile_literal(self, value, dialect):

        value = self.process_bind_param(value, dialect)

        if dialect.name == 'sqlite':
            return sa.expression.literal(value)

        if dialect.name == 'oracle':
            return sa.functions.to_date(
                value.strftime('%Y-%m-%d'), 
                'YYYY-MM-DD'
            )
        #if dialect.name == 'mysql':
        #    CONVERT_TZ('2020-01-03 07:24:25', '+00:00', '+01:00'
        #    return sa.functions.FROM_UNIXTIME((to_utc(value) - EPOCH).total_seconds())

        return sa.cast(value, self)

    def process_bind_param(self, value, dialect):

        #Wolfram side type system will ensure this is already compiled, no need for type checking
        if self.is_null(value):
            return None

        if isinstance(value, sa.Compilable):
            value = value.instance

        if dialect.name == 'sqlite':
            return (to_utc(value) - EPOCH).total_seconds()

        return value


class DateTime(Date):

    impl = NativeDateTime.impl

    def load_dialect_impl(self, dialect):

        if dialect.name == 'oracle':
            return sa.databases.oracle.TIMESTAMP(timezone=self.timezone)

        if dialect.name == 'sqlite':
            return Real()

        if dialect.name == 'mysql':
            if self.timezone:
                return sa.databases.mysql.TIMESTAMP()
            return sa.databases.mysql.DATETIME()

        if dialect.name == 'mssql':
            if self.timezone:
                return sa.databases.mssql.DATETIMEOFFSET()
            return sa.databases.mssql.DATETIME()

        return NativeDateTime(timezone=self.timezone)

    def compile_literal(self, value, dialect):

        value = self.process_bind_param(value, dialect)

        if dialect.name == 'sqlite':
            return sa.expression.literal(value)

        if dialect.name == 'oracle':
            #TIMEZONE?!?!
            if value.tzinfo:
                return sa.functions.TO_TIMESTAMP_TZ(
                    value.strftime('%Y-%m-%dT%H:%M:%S.%f%z'), 
                    'YYYY-MM-DD"T"HH24:MI:SS.FF9TZH:TZM'
                )                
            return sa.functions.TO_TIMESTAMP(
                value.strftime('%Y-%m-%dT%H:%M:%S.%f'), 
                'YYYY-MM-DD"T"HH24:MI:SS.FF9'
            )

        return sa.cast(value, self)

    def process_bind_param(self, value, dialect):

        if self.is_null(value):
            return None

        #Wolfram side type system will ensure this is already compiled, no need for type checking
        if isinstance(value, sa.Compilable):
            value = value.instance

        if dialect.name == 'sqlite':
            return (to_utc(value) - EPOCH).total_seconds()

        return value


class Time(Date):

    impl = NativeTime.impl

    def load_dialect_impl(self, dialect):

        if dialect.name == 'sqlite':
            return Real()

        if dialect.name == 'oracle':
            return Real()

        return NativeTime(timezone=self.timezone)

    def compile_literal(self, value, dialect):

        value = self.process_bind_param(value, dialect)

        if dialect.name == 'oracle':
            return sa.expression.literal(value)

        return sa.cast(value, self)

    def process_bind_param(self, value, dialect):

        #Wolfram side type system will ensure this is already compiled, no need for type checking

        if self.is_null(value):
            return None

        if isinstance(value, sa.Compilable):
            value = value.instance

        if dialect.name == 'sqlite' or dialect.name == 'oracle':

            if isinstance(value, datetime.time):
                value = to_utc(new_datetime(EPOCH, value, tzinfo=value.tzinfo))
                return (value - EPOCH).total_seconds()

        return value


class TimeQuantity(WolframType):

    impl = NativeTimeQuantity.impl

    def load_dialect_impl(self, dialect):

        if dialect.name == 'sqlite':
            return Real()

        return super(TimeQuantity, self).load_dialect_impl(dialect)
