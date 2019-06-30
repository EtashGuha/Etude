# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import datetime

from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import FunctionElement

from wolfram.sql.types import database_types as types
from wolframclient.utils.dispatch import Dispatch

dispatch = Dispatch()


@dispatch.dispatch(datetime.datetime)
def infer_type(instance,
               tz_type=types.wolfram.DateTime(True),
               default=types.wolfram.DateTime(False)):
    return instance.tzinfo and tz_type or default


@dispatch.dispatch(datetime.date)
def infer_type(instance, default=types.wolfram.Date()):
    return default


@dispatch.dispatch(datetime.time)
def infer_type(instance,
               tz_type=types.wolfram.Time(True),
               default=types.wolfram.Time(False)):
    return instance.tzinfo and tz_type or default


@dispatch.dispatch(datetime.timedelta)
def infer_type(instance, default=types.wolfram.TimeQuantity()):
    return default


class Compilable(FunctionElement):
    def __init__(self, instance):
        self.instance = instance
        self.type = dispatch(instance)
        super(Compilable, self).__init__()


@compiles(Compilable)
def compile(element, compiler, **kw):
    return compiler.process(
        element.type.compile_literal(element.instance, compiler.dialect))
