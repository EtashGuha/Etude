# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from wolfram.sql.types.info.wolfram import type_collection_for_dialect

supported_types = type_collection_for_dialect(
    dialect='postgresql',
    types={
        #'ARRAY':            'String',
        'BIGINT': 'Integer',
        #'BIT':              'ByteArray',
        'BOOLEAN': 'Boolean',
        'BYTEA': 'ByteArray',
        'CHAR': 'String',
        #'CIDR':             'String',
        'DATE': 'Date',
        #'DATERANGE':        'String',
        'DOUBLE_PRECISION': 'Real',
        #'ENUM': 'Choices',
        'FLOAT': 'Real',
        #'HSTORE':           'String',
        #'INET':             'String',
        #'INT4RANGE':        'String',
        #'INT8RANGE':        'String',
        'INTEGER': 'Integer',
        #'INTERVAL':         'String',
        #'JSON':             'String',
        #'JSONB':            'String',
        #'MACADDR':          'String',
        'NUMERIC': 'Decimal',
        #'NUMRANGE':         'String',
        #'OID':              'String',
        'REAL': 'Real',
        'SMALLINT': 'Integer',
        'TEXT': 'String',
        'TIME': 'Time',
        'TIMESTAMP': 'DateTime',
        #'TSRANGE':          'String',
        #'TSTZRANGE':        'String',
        #'UUID': 'UUID',
        'VARCHAR': 'String',
    })
