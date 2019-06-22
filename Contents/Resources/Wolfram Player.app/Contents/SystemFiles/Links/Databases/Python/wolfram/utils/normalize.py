# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import types
import uuid

from wolfram.sql import sqlalchemy as sa
from wolfram.sql.types.utils import get_type_info
from wolframclient.serializers import export
from wolframclient.utils import six
from wolframclient.utils.datastructures import Association
from wolframclient.utils.decorators import to_dict
from wolframclient.utils.encoding import force_text
from wolfram.sql import sqlalchemy as sa
from wolframclient.language import wl
from wolframclient.utils import six
from wolframclient.utils.functional import is_iterable
from wolframclient.serializers.encoder import wolfram_encoder
from wolframclient.utils import six
from wolframclient.language.expression import WLFunction, WLSymbol

def column_spec(table_name, column_name, schema = None):
    if schema:
        return '%s.%s.%s' % (schema, table_name, column_name)
    return '%s.%s' % (table_name, column_name)

def autodispatch(*types):
    def factory(func):
        def inner(serializer, o):
            return serializer.encode(func(o))
        wolfram_encoder.register(inner, types, replace_existing = True)
        return func
    return factory

@autodispatch(six.none_type)
def serialize_none(o):
    return WLSymbol('None')

@autodispatch(uuid.UUID)
def serialize_uuid(o):
    return force_text(o)

@autodispatch(sa.Engine)
def serialize_engine(o):
    return o.url

@autodispatch(sa.URL)
@to_dict
def serialize_url(url):
    yield "Backend", convert_bytes(url.drivername)
    yield "Username", convert_bytes(url.username)
    yield "Password", convert_bytes(url.password)
    yield "Host", convert_bytes(url.host)
    yield "Port", convert_bytes(url.port)
    yield "Name", convert_bytes(url.database)
    yield "Options", convert_bytes(url.query)

@autodispatch(sa.Sequence)
def serialize_sequence(o):
    #Sequence is the Default for columns with autoincrement, mssql is returning a sequence object
    #for now we just mark that as None in the wl side
    return None

@autodispatch(sa.Table)
@to_dict
def serialize_table(table, dialect=None):
    yield "PrimaryKey", table.primary_key
    yield "ForeignKeys", table.foreign_key_constraints
    yield "UniquenessConstraints", filter(lambda s: isinstance(s, sa.UniqueConstraint), table.constraints)
    yield "Indexes", table.indexes
    yield "Columns", Association(((column.name,
                                   serialize_column(column, dialect=dialect))
                                  for column in table.columns))
    yield "Schema", table.schema

@autodispatch(sa.PrimaryKeyConstraint)
@to_dict
def serialize_fk(constraint):
    yield "ConstraintName", constraint.name
    yield "Columns", (column.name for column in constraint.columns)

@autodispatch(sa.ForeignKeyConstraint)
@to_dict
def serialize_fk(constraint):
    yield "ConstraintName", constraint.name
    yield "FromColumns", (column.name for column in constraint.columns)
    yield "ToColumns", (fk.column.name for fk in constraint.elements)
    yield "ToTable", constraint.referred_table.name
    yield "OnUpdate", constraint.onupdate
    yield "OnDelete", constraint.ondelete
    yield "Initially", constraint.initially
    yield "Deferrable", constraint.deferrable

@autodispatch(sa.UniqueConstraint)
@to_dict
def serialize_unique(constraint):
    yield "ConstraintName", constraint.name
    yield "Columns", (column.name for column in constraint.columns)

@autodispatch(sa.Index)
@to_dict
def serialize_index(index):
    yield "IndexName", index.name
    yield "Columns", (column.name for column in index.columns)
    yield "Unique", bool(index.unique)

@autodispatch(sa.Column)
@to_dict
def serialize_column(column, dialect=None):
    #yield 'Indexed', bool(column.index)
    #yield 'Default', column.default
    #yield 'Unique', bool(column.unique)
    yield 'Nullable', column.nullable
    yield 'BackendType', serialize_type(column.type, dialect=dialect)
    if dialect:
        try:
            yield 'NativeTypeString', dialect.type_compiler.process(column.type)
        except sa.exceptions.CompileError:
            pass

@autodispatch(sa.MetaData)
@to_dict
def serialize_metadata(metadata, tables=None):
    for table in to_table(metadata, tables):
        yield table.name, serialize_table(
            table, dialect=metadata.bind and metadata.bind.dialect or None)


@autodispatch(sa.types.TypeEngine)
def serialize_type(type_, dialect=None):
    return get_type_info(type_, dialect=dialect).serialize_type(type_)


def convert_bytes(obj):
    if isinstance(obj, six.binary_type):
        return force_text(obj)
    return obj

def to_table(metadata, tables=None):

    if isinstance(tables, sa.Table):
        yield tables

    elif tables in (None, wl.All, wl.Automatic):
        for t in metadata.tables.values():
            yield t

    elif isinstance(tables, six.string_types):
        #the user might have specified a non existing table, what we should do?
        #for now i'm silently skipping it
        yield metadata.tables[tables]

    elif is_iterable(tables):
        for t in tables:
            for table in to_table(metadata, t):
                yield table
    else:
        raise NotImplementedError(
            'cannot handle tables using class %s' % tables.__class__)

def dumps(data, stream=None, **opts):
    return export(data, stream, **opts)
