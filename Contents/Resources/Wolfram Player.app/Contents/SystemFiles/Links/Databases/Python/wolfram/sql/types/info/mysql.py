# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import datetime

from wolfram.sql.types.info.wolfram import type_collection_for_dialect
from wolfram.sql.types.typeinfo import validators as v

supported_types = type_collection_for_dialect(
    dialect="mysql",
    types={
        "BIGINT":
        "Integer",
        "BINARY":
        "ByteArray",
        #"BIT":        "ByteArray",
        "BLOB":
        "ByteArray",
        "BOOLEAN":
        "Boolean",
        "CHAR":
        "String",
        "DATE":
        "Date",
        "TIMESTAMP": 
        dict(
            name='DateTime',
            validators=[v.fsp()],
            serializers=[v.timezone(attribute_processor = lambda val: True), v.fsp()],
            python_types=(datetime.datetime, ),
            extra_args = {'timezone': True}
        ),
        "DATETIME":
        dict(
            name='DateTime',
            validators=[v.fsp()],
            serializers=[v.timezone(attribute_processor = lambda val: False), v.fsp()],
            python_types=(datetime.datetime, ),
            extra_args = {'timezone': False}
            ),
        "DECIMAL":
        "Decimal",
        "DOUBLE":
        dict(
            name='Real',
            validators=[v.precision(), v.scale()],
            python_types=(float, ),
            extra_args={'asdecimal': False}),
        "ENUM": "String",
        "FLOAT":
        dict(
            name='Real',
            validators=[v.precision(), v.scale()],
            python_types=(float, ),
            extra_args={'asdecimal': False}),
        "INTEGER":
        "Integer",
        "LONGBLOB":
        "ByteArray",
        "LONGTEXT":
        "String",
        "MEDIUMBLOB":
        "ByteArray",
        "MEDIUMINT":
        "Integer",
        "MEDIUMTEXT":
        "String",
        "NCHAR":
        "String",
        "NUMERIC":
        "Decimal",
        "NVARCHAR":
        "String",
        "REAL":
        dict(
            name='Real',
            validators=[v.precision(), v.scale()],
            python_types=(float, ),
            extra_args={'asdecimal': False}),
        "SMALLINT":
        "Integer",
        "TEXT":
        "String",
        "TIME":
        dict(
            name='Time',
            validators=[v.timezone(required = False), v.fsp()],
            python_types=(datetime.time, )
        ),
        "TINYBLOB":
        "ByteArray",
        "TINYINT":
        "Integer",
        "TINYTEXT":
        "String",
        "VARBINARY":
        "ByteArray",
        "VARCHAR":
        "String",
        "YEAR":
        "Integer",
    })
