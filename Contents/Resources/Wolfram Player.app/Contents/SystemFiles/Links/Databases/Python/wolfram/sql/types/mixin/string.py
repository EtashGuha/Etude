# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from wolfram.sql import sqlalchemy as sa
from wolfram.sql.types.mixin.base import WolframType, construct_base_type
from wolframclient.utils import six


# String Types
class String(WolframType):

    impl = construct_base_type(sa.types.Unicode)

    def process_result_value(self, value, dialect):

        if self.is_null(value):
            return None

        if isinstance(value, six.string_types):
            return value

        return self.invalid_value(value)
