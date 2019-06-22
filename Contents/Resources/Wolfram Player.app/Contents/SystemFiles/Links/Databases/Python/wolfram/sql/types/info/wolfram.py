# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import datetime
import decimal
import uuid

from wolfram.sql import sqlalchemy as sa
from wolfram.sql.types.typeinfo import TypeCollection, TypeInfo
from wolfram.sql.types.typeinfo import validators as v
from wolframclient.utils import six


def type_collection_for_wolfram_types(**types):
    for key, value in types.items():
        value.setdefault('name', key)
        yield key, TypeInfo(**value)


supported_types = TypeCollection(
    type_collection_for_wolfram_types(
        Date=dict(
            base_class='wolfram.sql.types.mixin.date.Date',
            validators=[],
            python_types=(datetime.date, )),
        DateTime=dict(
            base_class='wolfram.sql.types.mixin.date.DateTime',
            validators=[v.timezone()],
            python_types=(datetime.datetime, )),
        Time=dict(
            base_class='wolfram.sql.types.mixin.date.Time',
            validators=[v.timezone(required = False)],
            python_types=(datetime.time, )),
        TimeQuantity = dict(
            base_class='wolfram.sql.types.mixin.date.TimeQuantity',
        ),
        #UUID=dict(
        #    base_class='wolfram.sql.types.mixin.uuid.UUID',
        #    python_types=(uuid.UUID, )),
        Integer=dict(
            base_class='wolfram.sql.types.mixin.numeric.Integer',
            python_types=six.integer_types),
        Decimal=dict(
            base_class='wolfram.sql.types.mixin.numeric.Decimal',
            validators=[v.precision(), v.scale()],
            python_types=(decimal.Decimal, )),
        Real=dict(
            base_class='wolfram.sql.types.mixin.numeric.Real',
            python_types=(float, )),
        ByteArray=dict(
            base_class='wolfram.sql.types.mixin.binary.ByteArray',
            validators=[v.binary_len()],
            python_types=(six.binary_type, )),
        Boolean=dict(
            base_class='wolfram.sql.types.mixin.boolean.Boolean',
            python_types=(bool, )),
        Choices=dict(
            base_class='wolfram.sql.types.mixin.choices.Choices',
            validators=[v.choices()],
            python_types=(six.text_type, six.binary_type)),
        String=dict(
            base_class='wolfram.sql.types.mixin.string.String',
            validators=[v.string_len(), v.collation()],
            python_types=(six.text_type, six.binary_type)),
        Unsupported=dict(
            base_class='wolfram.sql.types.mixin.unsupported.Unsupported',
        )
    ))


def _process_types(types, extra={}):
    for class_name, name_or_kwargs in types.items():
        if isinstance(name_or_kwargs, six.string_types):
            yield class_name, name_or_kwargs, extra
        else:
            yield class_name, name_or_kwargs.pop('name'), dict(
                extra, **name_or_kwargs)


def type_collection_for_dialect(dialect, types={}, **extra):

    defaults = {
        class_name: getattr(supported_types, name).copy(
            name=name,
            impl='sqlalchemy.dialects.%s.%s' % (dialect, class_name),
            **kwargs)
        for class_name, name, kwargs in _process_types(types, extra)
    }

    #registering all db overrides that are defined in sqlalchemy under dialect.colspecs
    for generic, specific in sa.databases[dialect].dialect.colspecs.items():
        for parent in specific.__mro__:
            try:
                defaults[generic.__name__] = defaults[parent.__name__]
                break
            except KeyError:
                pass

    return TypeCollection(defaults)
