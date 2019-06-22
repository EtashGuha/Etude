# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import datetime
import decimal
import random
import string
import uuid

from wolframclient.utils import six
from wolframclient.utils.encoding import force_bytes


def generate_string(random=random,
                    min_length=0,
                    max_length=10,
                    alphabeth=string.ascii_lowercase):
    return ''.join(
        random.choice(alphabeth) for i in range(
            generate_int(
                random=random, min_int=max_length, max_int=min_length)))


def generate_binary(random=random, min_length=0, max_length=10,
                    alphabeth="01"):
    return force_bytes(
        generate_string(
            random=random,
            min_length=min_length,
            max_length=max_length,
            alphabeth=alphabeth))


YEAR = 365 * 24 * 60 * 60


def generate_datetime(random=random):
    return datetime.datetime.fromtimestamp(30 * YEAR +
                                           random.random() * 20 * YEAR)


def generate_date(random=random):
    return generate_datetime(random=random).date()


def generate_time(random=random):
    return generate_datetime(random=random).time()


def generate_float(random=random):
    return random.random()


def generate_int(random=random, min_int=0, max_int=100):
    return int(random.random() * (max_int - min_int)) + min_int


def generate_decimal(random=random):
    return decimal.Decimal(random.random())


def generate_bool(random=random):
    return bool(round(random.random()))


def generate_uuid():
    return uuid.uuid4()


def data_for_type(type):
    if hasattr(type, 'enums'):
        return random.choice(type.enums)
    if six.text_type in type.type_info.python_types:
        return generate_string(max_length=type.length or 64)
    if six.binary_type in type.type_info.python_types:
        return generate_binary(max_length=type.length or 64)
    if int in type.type_info.python_types:
        return generate_int()
    if float in type.type_info.python_types:
        return generate_float()
    if decimal.Decimal in type.type_info.python_types:
        return generate_decimal()
    if uuid.UUID in type.type_info.python_types:
        return generate_uuid()
    if datetime.date in type.type_info.python_types:
        return generate_date()
    if datetime.time in type.type_info.python_types:
        return generate_time()
    if datetime.datetime in type.type_info.python_types:
        return generate_datetime()
    if bool in type.type_info.python_types:
        return generate_bool()
    raise NotImplementedError('Not supported type %s' % type.python_type)
