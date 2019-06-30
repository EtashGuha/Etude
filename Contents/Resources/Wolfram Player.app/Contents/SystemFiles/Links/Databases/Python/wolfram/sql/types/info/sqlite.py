# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from wolfram.sql.types.info.wolfram import type_collection_for_dialect

supported_types = type_collection_for_dialect(
    dialect="sqlite",
    types={
        #those are the actual type in sqlalchemy
        "INTEGER": "Integer",
        "TEXT": "String",
        "NUMERIC": "Decimal",
        "BLOB": "ByteArray",
        "REAL": "Real",
        "NullType": "ByteArray"
    })
