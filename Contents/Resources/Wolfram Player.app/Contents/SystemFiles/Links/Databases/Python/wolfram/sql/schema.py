# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from sqlalchemy import MetaData as SAMetaData
from sqlalchemy import Table as SATable
from sqlalchemy import util
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.base import ColumnCollection
from sqlalchemy.sql import quoted_name

from wolfram.sql import sqlalchemy as sa
from wolfram.sql.reflection import reflect
from wolframclient.utils.functional import iterate
from wolfram.utils.normalize import column_spec


#this is adding everywhere in sqlalchemy a method to extract a column from column name
#this method is creating a reverse lookup dictionary in the current ColumnCollection object.
#the reverse dict is then later used to quickly resolve column names


def get_column_from_name(self, column_name):

    #this is creating in the immutable dict a reverse lookup for column names
    #since the object is immutable by sql alchemy I cannot use setattr,
    #but i need to store the state in the __dict__ attribute.

    try:
        self.__dict__['__column_mapping__']
    except KeyError:
        self.__dict__['__column_mapping__'] = {
            column.name: column
            for column in self.values()
        }

    try:
        return self.__dict__['__column_mapping__'][column_name]
    except KeyError:
        raise KeyError('Cannot find column with name %s. Column names are %s' %
                       (column_name, ", ".join(
                           self.__dict__['__column_mapping__'].keys())))


ColumnCollection.get_column_from_name = get_column_from_name


class Table(SATable):
    def key_get(self):
        if not getattr(self, '_key', None):
            return super(Table, self).key
        return self._key

    def key_set(self, value):
        self._key = value

    key = property(key_get, key_set)


class MetaData(SAMetaData):
    def __init__(self, *args, **opts):

        self.table_aliases = opts.pop('table_aliases', None)
        self.column_aliases = opts.pop('column_aliases', None)

        super(MetaData, self).__init__(*args, **opts)

    reflect = reflect  #using custom reflect implementation

    @util.memoized_property
    def table_names(self):
        #we are using immutable dict because this cannot be changed anymore
        return {table.name: table for table in self.tables.values()}

    def get_table_from_name(self, table_name):
        try:
            return self.table_names[table_name]
        except KeyError:
            raise KeyError('Cannot find table with name %s. Table names are %s'
                           % (table_name, ", ".join(self.table_names.keys())))

    @classmethod
    def from_wl_spec(cls,
                     connection,
                     Tables={},
                     TableAliases={},
                     ColumnAliases={},
                     **opts):

        metadata = cls(
            table_aliases=TableAliases, column_aliases=ColumnAliases)

        #the schema is a dict, we should build tables now:
        for table_name, data in Tables.items():

            table_alias = metadata._resolve_table_alias(table_name)

            table = sa.Table(
                table_alias, metadata,
                *tuple(
                    metadata._parse_table_spec(
                        metadata, table_alias, data, connection=connection)), 
                schema = data.get('Schema', None) or None,
                quote = True
            )
            table.name = quoted_name(table_name, True)
            table.key = metadata._resolve_table_alias(table_name)

        return metadata

    def _resolve_column_alias(self, table_alias, column_name):
        try:
            return self.column_aliases[table_alias][column_name]
        except KeyError:
            return column_name

    def _resolve_table_alias(self, table_name):
        try:
            return self.table_aliases[table_name]
        except KeyError:
            return table_name

    def _parse_table_spec(self, metadata, table_alias, table, connection):
        for column_name, data in table.get("Columns", {}).items():
            yield self._parse_column(
                metadata=metadata,
                table_alias=table_alias,
                column_name=column_name,
                connection=connection,
                **data)

        for prop, func in (
            ("PrimaryKey", self._parse_primary_key),
            ("ForeignKeys", self._parse_foreign_key),
            ("UniquenessConstraints", self._parse_unique_columns),
            ("Indexes", self._parse_indexed_columns),
            ):
            value = table.get(prop, ())
            if isinstance(value, dict):
                value = value,

            for v in value:
                yield func(
                    metadata=metadata, 
                    table_alias=table_alias, 
                    Schema=table.get('Schema', None) or None,
                    **v
                )

    def _parse_primary_key(self,
                           metadata,
                           table_alias,
                           Columns=(),
                           ConstraintName=None, Schema = None):
        if Columns:
            return sa.PrimaryKeyConstraint(
                *(self._resolve_column_alias(table_alias, column_name)
                  for column_name in iterate(Columns)),
                name=ConstraintName)

    def _parse_foreign_key(self,
                           metadata,
                           table_alias,
                           ToTable,
                           FromColumns=(),
                           ToColumns=(),
                           OnUpdate=None,
                           OnDelete=None,
                           Initially=None,
                           Deferrable=None,
                           ConstraintName=None,
                           Schema=None):
        return sa.ForeignKeyConstraint([
            self._resolve_column_alias(table_alias, column_name)
            for column_name in iterate(FromColumns)
        ], [
            column_spec(self._resolve_table_alias(ToTable),
                       self._resolve_column_alias(
                           self._resolve_table_alias(ToTable), column_name), schema = Schema)
            for column_name in iterate(ToColumns)
        ],
                                       name=ConstraintName,
                                       onupdate=OnUpdate,
                                       ondelete=OnDelete,
                                       initially=Initially,
                                       deferrable=Deferrable)

    def _parse_unique_columns(self,
                           metadata,
                           table_alias,
                           Columns,
                           ConstraintName=None,
                           Schema=None
                           ):
        return sa.UniqueConstraint(*[
            self._resolve_column_alias(table_alias, column_name)
            for column_name in iterate(Columns)
        ], name = ConstraintName)

    def _parse_indexed_columns(self,
                           metadata,
                           table_alias,
                           Columns,
                           IndexName=None,
                           Unique=False,
                           Schema=None
                           ):
        return sa.Index(IndexName, *[
            self._resolve_column_alias(table_alias, column_name)
            for column_name in iterate(Columns)
        ], unique = Unique)

    def _parse_column(self,
                      metadata,
                      table_alias,
                      column_name,
                      connection=None,
                      Indexed=False,
                      Nullable=False,
                      PrimaryKey=False,
                      Default=None,
                      Unique=False,
                      BackendType=None,
                      **extra):

        return sa.Column(
            name=column_name,
            type_=self._parse_database_type(metadata, connection, BackendType),
            key=self._resolve_column_alias(table_alias, column_name),
            nullable=Nullable,
            default=Default,
            unique=Unique,
            index=Indexed,
            primary_key=PrimaryKey,
            quote=True
        )

    # API for schema declaration

    def _parse_database_type(self, metadata, connection, database_type=None):
        from wolfram.sql.types.utils import to_database_type

        return to_database_type(
            dialect=connection.create_engine().dialect,
            database_type=database_type,
        )
