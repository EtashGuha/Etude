# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import uuid

from wolfram.sql import sqlalchemy as sa
from wolfram.sql.types.mixin.base import WolframType, construct_base_type
from wolframclient.utils import six
from wolframclient.utils.functional import flatten


class Choices(WolframType):

    impl = construct_base_type(sa.types.Enum)

    def __init__(self, choices=(), _enums = (), name=None):
        return super(Choices, self).__init__(
            *flatten(choices, _enums), name=name or self.generate_name())

    def generate_name(self):
        # the proper thing to do is to use the column name, which is not available at this level
        return 'ENUM_%s' % uuid.uuid4()

    def process_result_value(self, value, dialect):

        if self.is_null(value):
            return None

        if isinstance(value, six.string_types):
            return value

        return self.invalid_value(value)
