# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from importlib import import_module

from wolframclient.utils.importutils import API

sqlalchemy = API(

    #database wrappers
    Column='sqlalchemy.Column',
    Table='wolfram.sql.schema.Table',
    MetaData='wolfram.sql.schema.MetaData',
    Sequence='sqlalchemy.Sequence',
    ForeignKeyConstraint='sqlalchemy.ForeignKeyConstraint',
    PrimaryKeyConstraint='sqlalchemy.PrimaryKeyConstraint',
    UniqueConstraint='sqlalchemy.UniqueConstraint',
    Index='sqlalchemy.Index',

    #custom imports that are only declaring custom dialects and patching sql alchemy internals
    Engine='sqlalchemy.engine.Engine',
    URL='sqlalchemy.engine.url.URL',
    create_connection='wolfram.sql.connection.create_connection',
    create_engine='wolfram.sql.connection.create_engine',
    engines='wolfram.sql.connection.engines',
    Connection='wolfram.sql.connection.Connection',

    #custom import of Compilable class for raw data
    Compilable='wolfram.sql.compilable.Compilable',
    databases=API(
        #we are only adding support for databases we have built in drivers.
        import_module,
        firebird='sqlalchemy.dialects.firebird',
        mssql='sqlalchemy.dialects.mssql',
        mysql='sqlalchemy.dialects.mysql',
        oracle='sqlalchemy.dialects.oracle',
        postgresql='sqlalchemy.dialects.postgresql',
        sqlite='sqlalchemy.dialects.sqlite',
        sybase='sqlalchemy.dialects.sybase',
    ),

    #functions
    types='sqlalchemy.types',
    expression='sqlalchemy.sql.expression',
    operators='sqlalchemy.sql.operators',
    select='sqlalchemy.sql.select',
    distinct='sqlalchemy.sql.distinct',
    functions='sqlalchemy.func',
    cast='wolfram.sql.functions.cast',

    #properties
    Properties='sqlalchemy.util.Properties',

    #exception
    exceptions='sqlalchemy.exc',
)
