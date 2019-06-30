# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import datetime
import decimal
import random
import string

import pytz

from wolfram.cli.dispatch import SimpleCommand
from wolfram.utils.normalize import dumps
from wolframclient.language import wl
from wolframclient.utils.datastructures import Association
from wolframclient.utils.decorators import to_dict
from wolframclient.utils.encoding import force_bytes
from wolframclient.utils.require import require
"""
Data can be formatted using

formatter =
 Composition[
  CellPrint[Cell[BoxData[#], "Output"]] &@
    DatabasesUtilities`Formatting`DBMakeFormattedBoxes[#, 100,
     Identity, {}] &,
  Style[#, ShowStringCharacters -> True] &
  ]

"""


def quantize(n=0, decimals=2, rounding=decimal.ROUND_HALF_EVEN):
    return decimal.Decimal(n).quantize(
        decimal.Decimal(1) / 10**decimals, rounding=rounding)


def random_real(f=0, s=1):
    return random.uniform(f, s)


def random_int(f=0, s=1):
    return random.randint(f, s)


def random_decimal(f=0, s=1, decimals=2):
    return quantize(random_real(f, s), decimals=decimals)


def random_datetime(year1=2018, year2=2020, tzinfo=pytz.utc):
    s = datetime.datetime(year=year1, month=1, day=1)
    s += datetime.timedelta(
        seconds=random_int(
            0,
            (datetime.datetime(year=year2, month=12, day=31) -
             s).total_seconds(),
        ))
    return tzinfo.localize(s)


def random_bytes(l1=8, l2=12):
    return force_bytes(''.join(
        random.choice(string.ascii_letters)
        for i in range(random_int(l1, l2))))


class Command(SimpleCommand):
    def add_arguments(self, parser):
        pass

    @to_dict
    @require('faker')
    def generate_data(self, persons=20, transactions=50):

        from faker import Faker
        fake = Faker()

        nations = (
            Association((('id', 'it'), ('name', "Italy"), ('currency',
                                                           'eur'))),
            Association((('id', 'fr'), ('name', "Spain"), ('currency',
                                                           'eur'))),
            Association((('id', 'us'), ('name', "USA"), ('currency', 'usd'))),
        )

        yield 'nation', nations
        yield 'person', (Association(
            (('id', i + 1),
             ('fullname',
              '%s %s' % (i % 2 and fake.first_name_male()
                         or fake.first_name_female(), fake.last_name())),
             ('address', fake.address().replace(
                 "\n", " - ")), ('sex', i % 2 and 'M' or 'F'), ('age',
                                                                random_int(
                                                                    18, 60)),
             ('weight', i % 2 and random_real(60, 90) or random_real(40, 65)),
             ('height', i % 2 and random_real(165, 190)
              or random_real(155, 180)), ('savings', random_decimal(
                  200, 5000)), ('licenseplate', fake.license_plate()),
             ('unemployed', not bool(i % 5)), ('datetime',
                                               date), ('date', date.date()),
             ('time', date.time()), ('unixtm',
                                     wl.UnixTime(date)), ('juliantm',
                                                          wl.JulianDate(date)),
             ('password', random_bytes()), ('friend', random_int(1, max(
                 1, i))), ('enemy', random_int(1, max(
                     1, i))), ('nation_id', random.choice(nations)['id'])))
                         for i, date in enumerate(
                             random_datetime() for i in range(persons)))
        yield 'transaction', (Association(
            (('id', i + 1), ('from',
                             random_int(1, persons)), ('to',
                                                       random_int(1, persons)),
             ('amount',
              random_decimal(100, 1000)), ('datetime', date), ('date',
                                                               date.date()),
             ('time', date.time()), ('nation_id', random.choice(nations)['id']),
             ('reason', '%s for %s. %s!'
              % (random.choice(('Payment', 'Transaction', 'Refound',
                                'Sending money', 'Thanks', 'Sending founds')),
                 random.choice(('ipad', 'holydays', 'goods', 'services',
                                'dinner', 'our flight', 'our honeymoon',
                                'computer', 'work you did', 'painting')),
                 random.choice(
                     ('Thanks', 'Cheers', 'Bests', 'See u soon', 'Kisses',
                      'Good work')))))) for i, date in enumerate(
                          random_datetime() for i in range(transactions)))

    def handle(self, **opts):

        self.print(dumps(self.generate_data(), indent=4))
