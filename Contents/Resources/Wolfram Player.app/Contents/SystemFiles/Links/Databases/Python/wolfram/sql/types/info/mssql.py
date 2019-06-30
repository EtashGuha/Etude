# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from wolfram.sql.types.info.wolfram import type_collection_for_dialect
from wolfram.sql.types.typeinfo import validators as v

import datetime

supported_types = type_collection_for_dialect(
    dialect='mssql',
    types={
        'SMALLINT': 'Integer',
        'XML': 'String',
        'SMALLDATETIME': 'DateTime',
        'VARCHAR': 'String',
        'TEXT': 'String',
        "DATETIME": 
        dict(
            name='DateTime',
            validators=[],
            serializers=[v.timezone(attribute_processor = lambda val: False)],
            python_types=(datetime.datetime, ),
            extra_args = {'timezone': False}
        ),
        "DATETIME2": 
        dict(
            name='DateTime',
            validators=[],
            serializers=[v.timezone(attribute_processor = lambda val: False)],
            python_types=(datetime.datetime, ),
            extra_args = {'timezone': False}
        ),
        "DATETIMEOFFSET": 
        dict(
            name='DateTime',
            validators=[],
            serializers=[v.timezone(attribute_processor = lambda val: True)],
            python_types=(datetime.datetime, ),
            extra_args = {'timezone': True}
        ),
        'CHAR': 'String',
        'TINYINT': 'Integer',
        'BIGINT': 'Integer',
        'TIME': 'Time',
        'DATE': 'Date',
        'INTEGER': 'Integer',
        'NVARCHAR': 'String',
        'NTEXT': 'String',
        'NCHAR': 'String',
        'BINARY': 'ByteArray',
        'VARBINARY': 'ByteArray',
        'IMAGE': 'ByteArray',
        'DECIMAL': 'Decimal',
        'NUMERIC': 'Decimal',
        'FLOAT': 'Real',
        'REAL': 'Real',
        'BIT': 'Boolean'
    })
