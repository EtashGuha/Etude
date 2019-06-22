# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from wolfram.sql import sqlalchemy as sa
from wolfram.sql.types.mixin.base import WolframType, construct_base_type
from wolframclient.utils import six


#Other types
class ByteArray(WolframType):

    #length – optional, a length for the column for use in DDL statements, for those binary types that accept a length, such as the MySQL BLOB type.

    impl = construct_base_type(sa.types.LargeBinary)

    def process_result_value(self, value, dialect):

        if self.is_null(value):
            return None

        if isinstance(value, six.binary_type):
            return value

        return self.invalid_value(value)
