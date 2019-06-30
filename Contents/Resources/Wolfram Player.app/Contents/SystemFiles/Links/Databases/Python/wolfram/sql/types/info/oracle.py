# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from wolfram.sql.types.info.wolfram import type_collection_for_dialect

supported_types = type_collection_for_dialect(
    dialect='oracle',
    types={
        'VARCHAR': 'String',
        'VARCHAR2': 'String',
        'NVARCHAR': 'String',
        'NVARCHAR2': 'String',
        'FLOAT': 'Real',
        'BINARY_FLOAT': 'Real',
        'BINARY_DOUBLE': 'Real',
        'CLOB': 'String',
        'TIMESTAMP': 'DateTime',
        'NUMBER': 'Decimal',
        'LONG': 'String',
        'CHAR': 'String',
        'RAW': 'ByteArray',
        'ROWID': 'String',
        'BLOB': 'ByteArray',
        'BFILE': 'ByteArray',
        'DATE': 'DateTime',
        'NCLOB': 'String',
        'INTEGER': 'Integer'
    })
