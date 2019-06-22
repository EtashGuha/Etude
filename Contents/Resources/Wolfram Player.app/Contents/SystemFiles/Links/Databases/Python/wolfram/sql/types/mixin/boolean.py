# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from wolfram.sql import sqlalchemy as sa
from wolfram.sql.types.mixin.base import WolframType, construct_base_type
from wolfram.sql.types.mixin.numeric import Integer


class NativeBoolean(WolframType):

    #Boolean Type is super strightforward, no arguments accepted

    #create_constraint – defaults to True. If the boolean is generated as an int/smallint, also create a CHECK constraint on the table that ensures 1 or 0 as a value.
    #name – if a CHECK constraint is generated, specify the name of the constraint.

    impl = construct_base_type(sa.types.Boolean)

    def process_result_value(self, value, dialect):

        if self.is_null(value):
            return None

        if isinstance(value, bool):
            return value

        if value in ('1', 'True', 'true'):
            return True

        if value in ('0', 'False', 'false'):
            return False

        return self.invalid_value(value)


class Boolean(WolframType):

    impl = construct_base_type(sa.types.Boolean)

    def load_dialect_impl(self, dialect):

        if dialect.name in ('sqlite', 'mssql'):

            return Integer()

        return NativeBoolean()
