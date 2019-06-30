# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import math
import operator
import re

from sqlalchemy.dialects.sqlite.pysqlite import SQLiteDialect_pysqlite

from wolfram.sql.engine.sqlite.mixin import SQLiteMixin

def function_factory(func, exceptions = ValueError):
    def inner(*args):
        if all(a is not None for a in args):
            try:
                return func(*args)
            except exceptions:
                return
    return inner

def regexp(string, pattern, insensitive):
    return bool(re.search(pattern, string, flags=insensitive and re.IGNORECASE or 0))

class Variance:
    def __init__(self):
        self.count = 0
        self.total = 0
        self.totalsq = 0

    def step(self, value):
        if value is not None: # Skipping NULLs
            self.count += 1
            self.total += value
            self.totalsq += value ** 2

    def finalize(self):
        if self.count < 2:
            return 0
        return self.totalsq / (self.count - 1) - self.total ** 2 / (self.count - 1)/self.count


class StandardDeviation(Variance):
    def finalize(self):
        return math.sqrt(Variance.finalize(self))


class SQLiteDialect(SQLiteMixin, SQLiteDialect_pysqlite):
    def on_connect(self):
        func = super(SQLiteDialect, self).on_connect()

        def inner(conn):
            if func:
                func(conn)

            conn.create_function("internal_regexp", 3, function_factory(regexp))

            conn.create_function("power", 2, function_factory(math.pow))
            conn.create_function("log", 1, function_factory(math.log))
            conn.create_function("floor", 1, function_factory(math.floor))
            conn.create_function("ceil", 1, function_factory(math.ceil))
            conn.create_function("sin", 1, function_factory(math.sin))
            conn.create_function("cos", 1, function_factory(math.cos))
            conn.create_function("tan", 1, function_factory(math.tan))
            conn.create_function("asin", 1, function_factory(math.asin))
            conn.create_function("acos", 1, function_factory(math.acos))
            conn.create_function("atan", 1, function_factory(math.atan))
            conn.create_function("atan2", 2, function_factory(math.atan2))

            conn.create_function("xor", 2, function_factory(operator.xor))


            conn.create_aggregate("variance", 1, Variance)
            conn.create_aggregate("stddev", 1, StandardDeviation)

        return inner


dialect = SQLiteDialect
