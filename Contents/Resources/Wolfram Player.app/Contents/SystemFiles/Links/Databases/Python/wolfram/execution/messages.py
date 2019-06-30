# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from wolfram.execution import logging
from wolfram.settings import settings
from wolfram.sql import sqlalchemy as sa
from wolframclient.language import wl
from wolframclient.language.exceptions import WolframLanguageException
from wolframclient.serializers.serializable import WLSerializable
from wolframclient.utils import six
from wolframclient.utils.datastructures import Settings
from wolframclient.utils.encoding import force_text
from wolframclient.utils.functional import iterate

DATABASE_FAILURE_TAG = "DatabaseFailure"
DATABASE_SUCCESS_TAG = "DatabaseSuccess"
DATABASE_VALIDATION_ERROR = "ValidationError"


def result_with_messages(result, *messages):
    if messages:
        return wl.CompoundExpression(*iterate(messages, (result, )))
    return result


# Failure generated from message class

class DatabaseExceptionDispatch(WolframLanguageException):


    def show_traceback(self):
        return settings.DEBUG

    def failure_tag(self):
        return DATABASE_FAILURE_TAG

    def failure_template(self):
        if isinstance(self.payload, sa.exceptions.SQLAlchemyError) and hasattr(self.payload, 'orig'):
            if not isinstance(self.payload, sa.exceptions.CompileError):
                return " ".join(map(force_text, self.payload.orig.args)).strip()
        return super(DatabaseExceptionDispatch, self).failure_template()

    def failure_meta(self):
        meta = super(DatabaseExceptionDispatch, self).failure_meta()

        if isinstance(self.payload, sa.exceptions.DBAPIError):
            if self.payload.statement:
                meta['SQLString'] = force_text(self.payload.statement).strip()
            if self.payload.params:
                meta['SQLParameters'] = self.payload.params

        return meta

class WolframLanguageMessageFailure(WolframLanguageException):
    def __init__(self,
                 message,
                 exec_info=None,
                 failure_code=None,
                 failure_tag=None,
                 show_traceback=False):

        super(WolframLanguageMessageFailure, self).__init__(
            message, exec_info=exec_info)

        self._show_traceback = show_traceback
        self._failure_tag = failure_tag
        self._failure_code = failure_code

    def show_traceback(self):
        return self._show_traceback

    def failure_tag(self):
        return self._failure_tag or DATABASE_FAILURE_TAG

    def failure_template(self):
        return self.payload.to_wl_message()

    def failure_parameters(self):
        return self.payload.args

    def failure_code(self):
        return self._failure_code or DATABASE_VALIDATION_ERROR

class WolframLanguageMessageSuccess(WolframLanguageMessageFailure):

    def to_wl(self, *args, **opts):
        return wl.Success(
            self.failure_tag(),
            wl.Association(*(wl.RuleDelayed(key, value)
                             for key, value in self.failure_meta().items())))

    def failure_code(self):
        return self._failure_code

    def failure_tag(self):
        return self._failure_tag or DATABASE_SUCCESS_TAG

# Message factory class

# those messages needs to be in sync with Messages.m
# python code is allowed to call only messages statically defined here

# sample usage:
# from wolfram.execution import messages
# messages.invalidtype(arg1, arg2, arg3)


class Message(WLSerializable):

    default_symbol_factory = wl
    exception_class = WolframLanguageMessageFailure
    success_class = WolframLanguageMessageSuccess

    def __init__(self, symbol, messagename, *args):
        if isinstance(symbol, six.string_types):
            self.symbol = getattr(self.default_symbol_factory, symbol)
        else:
            self.symbol = symbol

        self.messagename = messagename
        self.args = args

    def to_wl_message(self):
        return wl.MessageName(self.symbol, self.messagename)

    def to_wl(self, *args, **opts):
        return wl.Message(self.to_wl_message(), *self.args)

    def as_exception(self, **opts):
        return self.exception_class(self, **opts)

    def as_success(self, **opts):
        return self.success_class(self, **opts)

    def send_message(self):
        return logging.send_message(self)


class MessageFactory(object):

    message_class = Message

    def __init__(self,
                 reference,
                 argcount=0,
                 show_traceback=False,
                 failure_tag=None,
                 failure_code=None):

        self.symbol, self.messagename = reference.split('::')

        self.argcount = argcount
        self.failure_tag = failure_tag
        self.show_traceback = show_traceback
        self.failure_code = failure_code

    def __call__(self, *args):
        assert len(args) == self.argcount
        return self.message_class(self.symbol, self.messagename, *args)

    def as_exception(self, *args):
        return self(*args).as_exception(
            failure_tag=self.failure_tag,
            failure_code=self.failure_code,
            show_traceback=self.show_traceback,
        )

    def as_success(self, *args):
        return self(*args).as_success(
            failure_tag=self.failure_tag,
            failure_code=self.failure_code,
            show_traceback=self.show_traceback,
        )

    def send_message(self, *args):
        return self(*args).send_message()

    def __repr__(self):
        return '<%s %s::%s>' % (self.__class__.__name__, self.symbol,
                                self.messagename)


messages = Settings(
    #db inspection
    inspection_cannot_infer_type=MessageFactory(
        'RelationalDatabase::nvldtype', argcount=3),
    inspection_drops_fk=MessageFactory(
        'RelationalDatabase::inspfk', argcount=2),
    inspection_missing_table=MessageFactory(
        'RelationalDatabase::inspmtb', argcount=1),
    inspection_shadow_table=MessageFactory(
        'RelationalDatabase::insptb', argcount=2),
    #unsupported sql operation
    unsupported_backend=MessageFactory(
        'RelationalDatabase::nvldbackend',
        argcount=2,
        failure_code="UnsupportedOperation"),
    #type validation
    type_is_invalid=MessageFactory(
        'RelationalDatabase::typenvld',
        argcount=1,
    ),
    type_require_argument=MessageFactory(
        'RelationalDatabase::typerequired',
        argcount=1,
    ),
    type_require_int_argument=MessageFactory(
        'RelationalDatabase::typeint',
        argcount=2,
    ),
    type_require_string_argument=MessageFactory(
        'RelationalDatabase::typestring',
        argcount=2,
    ),
    type_require_choices_argument=MessageFactory(
        'RelationalDatabase::typechoices',
        argcount=2,
    ),
    type_require_boolean_argument=MessageFactory(
        'RelationalDatabase::typeboolean',
        argcount=2,
    ),
    type_argument_count=MessageFactory(
        'RelationalDatabase::typeargcount',
        argcount=3,
    ),
    #connection
    connection_done = MessageFactory(
        'DatabaseReference::conn',
    ),    
    connection_done_already = MessageFactory(
        'DatabaseReference::aconn',
    ),
    disconnection_done = MessageFactory(
        'DatabaseReference::disc',
    ),
    disconnection_done_already = MessageFactory(
        'DatabaseReference::adisc',
    ),
    sqlite_missing_file = MessageFactory(
        'DatabaseReference::nvldfile',
        argcount = 1
    )
)
