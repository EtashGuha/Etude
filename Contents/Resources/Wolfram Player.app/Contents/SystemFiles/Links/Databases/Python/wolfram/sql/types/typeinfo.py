# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from collections import OrderedDict
from itertools import groupby, product
from operator import attrgetter

from wolfram.execution.messages import messages
from wolframclient.language import wl
from wolframclient.serializers.serializable import WLSerializable
from wolframclient.settings import NOT_PROVIDED
from wolframclient.utils import six
from wolframclient.utils.datastructures import Settings
from wolframclient.utils.encoding import force_bytes, force_text
from wolframclient.utils.functional import first, is_iterable, iterate, identity
from wolframclient.utils.importutils import safe_import_string


class ArgumentValidator(WLSerializable):
    def __init__(self,
                 validator,
                 wl_test,
                 error,
                 attribute,
                 name=None,
                 parameter=None,
                 wl_parameter=None,
                 default=None,
                 required=False,
                 attribute_processor = identity,
                 valid_tests=[],
                 invalid_tests=[]):
        self.wl_test = wl_test
        self.validator = validator
        self.error = error
        self.valid_tests = valid_tests
        self.invalid_tests = invalid_tests
        self.default = default
        self.required = required
        self.name = name

        #used for calling the class and reconstructing it
        self.attribute_processor = attribute_processor
        self.attribute = attribute
        self.parameter = parameter or attribute
        self.wl_parameter = wl_parameter or parameter or attribute.title()

    def to_wl(self, *args, **opts):
        return {
            'Test': self.wl_test,
            'Message': wl.Function(self.error(wl.Slot())),
            'Default': self.default,
            'Required': self.required,
            'Parameter': self.wl_parameter
        }

    def get_attribute(self, obj):
        return self.attribute_processor(getattr(obj, self.attribute, None))

    def __call__(self, arg):

        if self.required and arg in (None, NOT_PROVIDED):
            raise messages.type_require_argument.as_exception(self.name)

        if arg is NOT_PROVIDED:
            return self.default

        if self.validator(arg):
            return arg

        raise self.error(arg)

    def __repr__(self):
        return '<Argument %s>' % self.parameter


def generate_integer(name, min_value=0, max_value=None, **opts):
    return ArgumentValidator(
        name  = name,
        wl_test = wl.Function(
            wl.And(
                wl.IntegerQ(wl.Slot()),
                min_value is None and True or wl.GreaterEqual(wl.Slot(), min_value),
                max_value is None and True or wl.LessEqual(wl.Slot(),    max_value),
            )
        ),
        validator     = lambda value:
            isinstance(value, six.integer_types) and
                (min_value is None and True or value >= min_value) and
                (max_value is None and True or value <= max_value),
        error         = lambda value: messages.type_require_int_argument.as_exception(name, value),
        **opts
    )


def generate_string(name, **opts):
    return ArgumentValidator(
        name      = name,
        wl_test   = wl.StringQ,
        validator = lambda value: isinstance(value, six.string_types),
        error     = lambda value: messages.type_require_string_argument.as_exception(name, value),
        **opts
    )


def generate_list(name, **opts):
    return ArgumentValidator(
        name      = name,
        wl_test   = wl.ListQ,
        validator = lambda value: is_iterable(value),
        error     = lambda value: messages.type_require_choices_argument.as_exception(name, value),
        **opts
    )


def generate_bool(name, **opts):
    return ArgumentValidator(
        name      = name,
        wl_test   = wl.BooleanQ,
        validator = lambda value: value in (True, False, None),
        error     = lambda value: messages.type_require_boolean_argument.as_exception(name, value),
        **opts
    )


def partial(func, **opts):
    #custom partial that implementes defaults
    return lambda **extra: func(**dict(opts, **extra))


validators = Settings(
    scale=partial(
        generate_integer,
        name='Scale',
        valid_tests=[4, 2],
        invalid_tests=['something'],
        attribute='scale',
        required=False,
        default=2),
    precision=partial(
        generate_integer,
        name='Precision',
        valid_tests=[10, 5],
        invalid_tests=['something'],
        attribute='precision',
        required=False,
        default=10,
    ),
    string_len=partial(
        generate_integer,
        name='String length',
        valid_tests=[10, 20],
        invalid_tests=['something'],
        attribute='length',
        default=64),
    binary_len=partial(
        generate_integer,
        name='Binary length',
        valid_tests=[10, 20],
        invalid_tests=['something'],
        attribute='length',
        default=64),
    collation=partial(
        generate_string,
        name='Collation',
        valid_tests=[],
        invalid_tests=[1],
        attribute='collation',
    ),
    timezone=partial(
        generate_bool,
        name='TimeZone',
        valid_tests=[0, 1],
        required=True,
        invalid_tests=['something'],
        wl_parameter='TimeZone',
        attribute='timezone',
        attribute_processor=bool,
        default=False
    ),
    fsp=partial(
        generate_integer,
        name='Fractional second precision',
        valid_tests=[0, 6],
        invalid_tests=['0'],
        attribute='fsp',
        max_value=6,
    ),
    choices=partial(
        generate_list,
        name='choices',
        valid_tests=[["book", "pen", "agenda"], ["apple", "orange", "banana"]],
        invalid_tests=['a', 1],
        wl_parameter='Choices',
        parameter='choices',
        attribute='_enumerated_values',
        required=True))


class TypeCollection(Settings):
    def generate_possible_tests(self):
        for t in self.values():
            for sub in t.generate_possible_tests():
                yield sub

    def __repr__(self):
        return '<TypeCollection: %s>' % list(self.keys())


class TypeInfo(WLSerializable):
    def __init__(self,
                 name,
                 base_class,
                 impl=None,
                 validators=(),
                 serializers=None,
                 python_types=(),
                 extra_args={},
                 wl_name=None,
                 type_attrs={},
                 impl_attrs={}):
        self.name = name
        self.wl_name = wl_name or name
        self.base_class = base_class
        self.impl = impl
        self.impl_attrs = impl_attrs
        self.type_attrs = type_attrs
        self.python_types = python_types
        self.validators = validators
        self.extra_args = extra_args

        if serializers is None:
            self.serializers = self.validators
        else:
            self.serializers = serializers

    def copy(self, **opts):
        kwargs = {
            attr: getattr(self, attr)
            for attr in ('name', 'wl_name', 'base_class', 'impl', 'validators',
                         'python_types', 'extra_args')
        }
        kwargs.update(opts)
        return self.__class__(**kwargs)

    def to_wl(self, *args, **opts):
        return {
            'Type': force_text(self.wl_name),
            'Arguments': OrderedDict(
                (v.wl_parameter, v) for v in self.validators)
        }

    @property
    def db_type(self):
        try:
            return self._imported_class
        except AttributeError:

            attrs = {'type_info': self}

            if self.impl:
                attrs['impl'] = safe_import_string(self.impl)

                if self.impl_attrs:
                    attrs['impl'] = type(attrs['impl'].__name__,
                                         (attrs['impl'], ), self.impl_attrs)

            attrs.update(self.type_attrs)

            self._imported_class = type(
                six.PY2 and force_bytes(self.name) or self.name,
                (safe_import_string(self.base_class), ), attrs)
            return self._imported_class

    def validate(self, *args):

        kwargs = {}
        kwargs.update(self.extra_args)

        delta = len(self.validators) - len(args)

        if delta < 0:
            raise messages.type_argument_count(
                force_text(self.name), len(args),
                len(self.validators)).as_exception()

        for arg, validator in zip(
                iterate(args, (NOT_PROVIDED for i in range(delta))),
                self.validators):

            kwargs[validator.parameter] = validator(arg)

        return kwargs

    def generate_possible_tests(self):
        for values in self.valid_tests():
            yield self, values

    def valid_tests(self):

        if not self.validators or all(not v.required for v in self.validators):
            yield ()

        for sub in product(*map(attrgetter('valid_tests'), self.validators)):
            yield sub

    def serialize_type(self, type_):
        #this is used by the WL serializer to create the WL symbolic version of this type:
        serialized = tuple(
            iterate(
                force_text(self.name),
                (val.get_attribute(type_) for val in self.serializers)))

        #now we do a groupby over the reversed in list in order to check if we can do a shorter reppresentation without leading None

        grouper, values = first(groupby(reversed(serialized)))

        if grouper == None:
            serialized = serialized[:-len(tuple(values))]

        if len(serialized) == 1:
            return first(serialized)

        return serialized

    def __repr__(self):
        return '<TypeInfo for %s>' % self.name

    def __call__(self, *args):
        return self.db_type(**self.validate(*args))
