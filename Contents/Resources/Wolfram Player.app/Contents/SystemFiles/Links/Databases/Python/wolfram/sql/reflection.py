# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from itertools import groupby

from sqlalchemy import inspect

from wolfram.execution.messages import messages
from wolfram.sql import sqlalchemy as sa
from wolfram.utils.normalize import column_spec
from wolframclient.utils.decorators import decorate, to_dict
from wolframclient.utils.functional import first, iterate, last

SYSTEM_SCHEMA_EXPLORE = {
    'postgresql': lambda inspector: frozenset(inspector.get_schema_names()).difference(('information_schema', ))
}

@to_dict
def regroup_tables(pairs):
    for table, schemas in groupby(sorted(pairs, key=first), key=first):
        schemas = tuple(map(last, schemas))
        yield table, first(schemas)
        if len(schemas) > 1:
            messages.inspection_shadow_table.send_message(
                table, ', '.join(schemas))


@decorate(regroup_tables)
def get_all_tables(inspector, metadata, only=None, schema=None):

    if schema is None:
        if metadata.bind.dialect.name in SYSTEM_SCHEMA_EXPLORE:
            schemas = SYSTEM_SCHEMA_EXPLORE[metadata.bind.dialect.name](inspector)
        else:
            schemas = (None, )
    else:
        schemas = iterate(schema)

    for schema in schemas:
        for table in inspector.get_table_names(schema=schema):
            if only is None or table in only:
                yield table, schema


def to_column(dialect,
              primary_keys,
              name,
              type,
              primary_key=False,
              sequence=None,
              **opts):

    return sa.Column(
        name=name,
        type_=type,
        primary_key=primary_key or name in primary_keys,
        **opts)


def to_fk(name,
          constrained_columns,
          referred_schema,
          referred_table,
          referred_columns,
          from_table,
          schema,
          options={},
          include_related_tables=False,
          remaining_tables=(),
          all_tables=()):

    if referred_table in all_tables:
        if all_tables[referred_table] == referred_schema:
            return sa.ForeignKeyConstraint(
                constrained_columns,
                tuple(
                    column_spec(referred_table, c, schema=schema)
                    for c in referred_columns),
                name=name,
                referred_schema=referred_schema,
                **options)
        else:
            #this is a bug and needs to be handled, the foreign key references a table with the same name
            #but on another schema, for now we are dropping duplicated names
            pass

    elif include_related_tables:
        all_tables[referred_table] = schema
        remaining_tables.add(referred_table)
    else:
        messages.inspection_drops_fk.send_message(from_table, referred_table)


def to_unique(column_names, **extra):
    return sa.UniqueConstraint(*column_names, **extra)


def to_index(column_names, name=None, type=None, **extra):
    return sa.Index(name, *column_names, **extra)


def reflect(metadata, only=None, schema=None, include_related_tables=False):
    #None means all tables

    insp = inspect(metadata.bind)

    all_tables = get_all_tables(
        insp, only=only, schema=schema, metadata=metadata)

    if only:
        for missing in frozenset(iterate(only)).difference(all_tables):
            messages.inspection_missing_table.send_message(missing)

    remaining_tables = set(all_tables)

    while remaining_tables:
        table = remaining_tables.pop()
        schema = all_tables[table]

        if not metadata.bind.dialect.name == 'sqlite':
            primary_keys = insp.get_primary_keys(table, schema=schema)
        else:
            primary_keys = ()

        sa.Table(
            table,
            metadata,
            *(to_column(
                dialect=metadata.bind.dialect,
                primary_keys=primary_keys,
                **meta) for meta in insp.get_columns(table, schema=schema)),
            *(to_unique(**meta)
              for meta in insp.get_unique_constraints(table, schema=schema)),
            *(to_index(**meta)
              for meta in insp.get_indexes(table, schema=schema)),
            *filter(lambda obj: not obj is None,
                    (to_fk(
                        remaining_tables=remaining_tables,
                        all_tables=all_tables,
                        from_table=table,
                        include_related_tables=include_related_tables,
                        schema=schema,
                        **meta)
                     for meta in insp.get_foreign_keys(table, schema=schema))),
            schema=schema)
