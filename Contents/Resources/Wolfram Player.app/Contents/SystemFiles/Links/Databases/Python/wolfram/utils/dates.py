# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import datetime
import decimal
import re

from wolframclient.utils import six
from wolframclient.utils.encoding import force_text
from wolframclient.utils.importutils import API

ABSOLUTE_OFFSET_RE = re.compile(
    '(?P<sign>[+-]?)(?P<hour>[0-9]{1,2})(:(?P<minutes>[0-9]{2}))?')

pytz = API(
    FixedOffset='pytz.FixedOffset',
    timezone='pytz.timezone',
    utc='pytz.utc',
    UnknownTimeZoneError='pytz.UnknownTimeZoneError')


def new_date(date):
    return datetime.date(
        year=date.year,
        month=date.month,
        day=date.day,
    )


def new_time(time, tzinfo=None):
    return datetime.time(
        hour=time.hour,
        minute=time.minute,
        second=time.second,
        microsecond=time.microsecond,
        tzinfo=to_timezone(tzinfo))


def new_datetime(date, time=None, tzinfo=None):

    copy = datetime.datetime(
        year=date.year,
        month=date.month,
        day=date.day,
        hour=time and time.hour or 0,
        minute=time and time.minute or 0,
        second=time and time.second or 0,
        microsecond=time and time.microsecond or 0,
    )

    if tzinfo is not None:
        return to_timezone(tzinfo).localize(copy)
    return copy


def _offset_to_timezone(sign, hour, minutes):
    return pytz.FixedOffset(
        (sign == '-' and -1 or 1) * (int(hour) * 60 + int(minutes or 0)))


def to_timezone(value, allow_string_offset=False):
    if isinstance(value, (six.integer_types, float, decimal.Decimal)):
        tzinfo = pytz.FixedOffset(value * 60)
        if six.JYTHON:
            #jython source code has a bug that is preventing to take the right path if the tzname method is returning None
            #to prevent this we are returning an artificial name using the offset value
            tzinfo.tzname = lambda *args, **opts: force_text(value)
        return tzinfo
    if isinstance(value, datetime.tzinfo):
        return value
    if isinstance(value, six.string_types):
        if allow_string_offset:
            m = ABSOLUTE_OFFSET_RE.match(value)
            if m:
                return _offset_to_timezone(**m.groupdict())

        return pytz.timezone(value)
    if value is None:
        return None
    if isinstance(value, (datetime.datetime, datetime.time)):
        return to_timezone(value.tzinfo)
    raise TypeError('cannot cast timezone offset from %s' % value)
