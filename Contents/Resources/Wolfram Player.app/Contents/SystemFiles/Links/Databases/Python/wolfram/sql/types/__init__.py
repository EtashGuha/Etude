# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from wolframclient.utils.importutils import API

database_types = API(
    mssql='wolfram.sql.types.info.mssql.supported_types',
    mysql='wolfram.sql.types.info.mysql.supported_types',
    oracle='wolfram.sql.types.info.oracle.supported_types',
    postgresql='wolfram.sql.types.info.postgresql.supported_types',
    sqlite='wolfram.sql.types.info.sqlite.supported_types',
    wolfram='wolfram.sql.types.info.wolfram.supported_types',
)
