# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import base64
import datetime
import decimal
import math
from functools import reduce, wraps
from itertools import chain

from wolfram.execution import logging
from wolfram.execution.messages import (DatabaseExceptionDispatch, messages,
                                        result_with_messages)
from wolfram.sql.types.utils import get_type_info
from wolfram.settings import settings
from wolfram.sql import sqlalchemy as sa
from wolfram.utils.normalize import to_table
from wolfram.sql.render import render_query
from wolfram.sql.types import database_types as types
from wolfram.utils.dates import new_datetime, new_time, to_timezone
from wolfram.utils.normalize import dumps
from wolframclient.deserializers import binary_deserialize
from wolframclient.language import wl
from wolframclient.deserializers.wxf.wxfconsumer import WXFConsumer
from wolframclient.language.decorators import safe_wl_execute
from wolframclient.language.expression import WLFunction, WLSymbol
from wolframclient.utils import six
from wolframclient.utils.datastructures import Association
from wolframclient.utils.decorators import to_dict
from wolframclient.utils.dispatch import Dispatch
from wolframclient.utils.encoding import force_text
from wolframclient.utils.functional import first, iterate, last, partition

evaluate = Dispatch()

@evaluate.dispatch(dict)
def evaluate_expr(self, expression):
    return expression.__class__(
        (key, self.evaluate(value)) for key, value in expression.items())

@evaluate.dispatch(six.iterable_types)
def evaluate_expr(self, expression):
    return tuple(self.evaluate(e) for e in expression)

@evaluate.dispatch(WLFunction)
def evaluate_expr(self, expression):

    if isinstance(expression.head, WLSymbol):
        attribute = last(expression.head.name.split('`'))
    elif isinstance(expression.head, six.string_types):
        attribute = expression.head
    else:
        raise ValueError('Unable to evaluate %s' % expression)
    try:
        function = getattr(self, attribute)
    except AttributeError:
        raise ValueError('Unable to evaluate %s' % attribute)

    return function(*expression.args)

@evaluate.dispatch()
def evaluate_expr(self, expression):
    return expression


def is_data_type(expr,
                 types=tuple(
                     chain(
                         six.string_types,
                         six.integer_types,
                         six.buffer_types,
                         (
                             datetime.date,
                             datetime.time,
                             float,
                             decimal.Decimal,
                             bool,
                             six.none_type,
                         ),
                     ))):
    return isinstance(expr, types)


def unsupported(operation, *backends):
    def outer(function):
        @wraps(function)
        def inner(evaluator, *args, **opts):

            if evaluator.backend_name() in backends:
                raise messages.unsupported_backend.as_exception(
                    operation, evaluator.backend_name())

            return function(evaluator, *args, **opts)

        return inner

    return outer


def backend_based(function):
    def inner(evaluator, *args, **opts):

        method = getattr(
            evaluator, '%s_%s' % (function.__name__, evaluator.backend_name()),
            None)

        if method:
            return method(*args, **opts)

        return function(evaluator, *args, **opts)

    return inner

def prevent_implicit_joins(selectable):
    result = selectable
    result._froms[0]._hide_froms = [f for f in result._froms if f != result._from_obj[0]]
    return result.correlate_except(None)

class DatabasesWXFConsumer(WXFConsumer):
    
    BUILTIN_SYMBOL = {
        'True': True,
        'False': False,
        'None': None,
        'Null': None,
        'Pi': math.pi,
        'Indeterminate': float('NaN')
    }

    def consume_symbol(self, current_token, tokens, **kwargs):
        """Consume a :class:`~wolframclient.deserializers.wxf.wxfparser.WXFToken` of type *symbol* as a :class:`~wolframclient.language.expression.WLSymbol`"""
        try:
            return self.BUILTIN_SYMBOL[current_token.data]
        except KeyError:
            return WLSymbol(current_token.data)

class WolframLanguageEvaluator(object):

    #wolfram language evaluator used by DBRunPython

    #the idea is very simple you pass to DBRunPython something like DatabaseQuery[args] and it will call a method of the evaluator called DatabaseQuery(self, args)

    #everything that is CamelCased is something that is used on the wolfram language side.
    #everything that is lower_cased is something used internally

    #WolframLanguageEvaluator implements a context manager logic
    #we want to make sure that you can manually close all pending operations by calling close, or you use with in order to auto-close all pending operations that must be closed

    def __init__(self):
        self.metadata = None
        self.path = None
        self.connection = None
        self.target_format = 'wl'
        self.messages = logging.new_accumulator()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def close(self):
        #if self.connection:
        #    self.connection.close()
        self.messages.remove_logger()

    evaluate = evaluate.as_method()

    def evaluate_wxf_with_messages(self, wxf):
        result = self.evaluate(binary_deserialize(wxf, consumer = DatabasesWXFConsumer()))
        return result_with_messages(result, *self.messages)

    def backend_name(self):
        return force_text(self.connection.create_engine().dialect.name)

    def safe_wxf_evaluate_with_messages(
            self,
            expr,
            is_base64=False,
            target_format="wxf",
            **export_opts):

        return safe_wl_execute(
            self.evaluate_wxf_with_messages,
            args=(expr, ),
            export_opts=dict(
                target_kernel_version = settings.TARGET_KERNEL_VERSION, 
                target_format = target_format, 
                **export_opts
            ),
            exception_class=DatabaseExceptionDispatch)

    #specific implementation of SQLAlchemy

    def to_connection(self, connection):
        if isinstance(connection, sa.Connection):
            return connection
        elif connection in (None, wl.Automatic):
            #this is creating in memory sqlite
            return sa.create_connection('sqlite://')

        #this version of create engine accepts multiple args and normalize them
        return sa.create_connection(connection)

    def to_metadata(self, schema=wl.Automatic):

        if isinstance(schema, sa.MetaData):
            return schema

        elif schema in (wl.All, wl.Automatic):

            metadata = sa.MetaData()
            metadata.reflect(bind=self.connection.create_engine())

            return metadata

        elif schema is None:
            return sa.MetaData(bind=self.connection.create_engine())

        elif isinstance(schema, dict):
            return sa.MetaData.from_wl_spec(
                connection=self.connection, **schema)
        else:
            raise NotImplementedError(
                'cannot build metadata using class %s' % schema.__class__)

    # API for schema declaration
    def parse_env(self, metadata=None, connection=None, path=None, **options):

        yield 'connection', self.to_connection(connection)
        yield 'metadata', self.to_metadata(metadata)
        yield 'path', path

        for key, value in options.items():
            yield key, self.evaluate(value)

    def export_to_file(self, expr):
        if isinstance(self.path, six.string_types):
            return wl.File(
                dumps(expr, self.path, target_format=self.target_format))
        return expr

    def serialize_sql_object(self):
        return wl.Databases.Schema.DBCreateValidDatabase(
            self.export_to_file(Association(Tables=self.metadata)),
            wl.DatabaseReference(self.connection.create_engine())
        )

    def wl_transpose(self, keys, data):
        if data.rowcount == 0:
            return ()
        return wl.Transpose(
            wl.AssociationThread(map(force_text, keys), wl.Transpose(data)),
            AllowedHeads=wl.All)

    #API to get database informations

    def DatabaseInformation(self):
        #lazy version
        eng = self.connection.create_engine()
        try:
            return eng._db_info
        except AttributeError:
            eng._db_info = self.DatabaseInformationGet()
            return eng._db_info


    #API to manually manage connection

    def DatabaseReferences(self):
        return sa.engines

    def DatabaseConnected(self):
        return self.connection.is_connected()

    def DatabaseDisconnect(self):
        if self.connection.is_connected():
            self.connection.disconnect()
            return messages.disconnection_done.as_success()
        return messages.disconnection_done_already.as_success()

    def DatabaseConnect(self):
        if not self.connection.is_connected():
            self.connection.connect()
            return messages.connection_done.as_success()
        return messages.connection_done_already.as_success()

    #API for operations

    def DatabaseCreate(self, tables=None):
        self.metadata.create_all(
            tables=to_table(self.metadata, tables),
            bind=self.connection.create_engine())
        return self.serialize_sql_object()

    def DatabaseDrop(self, tables=None):

        select_tables = tuple(to_table(self.metadata, tables))

        self.metadata.drop_all(
            tables=select_tables, bind=self.connection.create_engine())

        for t in select_tables:
            # manually calling _remove_table because there is a bug with aliasing that i reported.
            # https://bitbucket.org/zzzeek/sqlalchemy/issues/4108/metadata-remove-should-use-tablekey
            # if something changes in future versions our test suite should catch this problem.
            # hopefully one day we will be using the official method metadata.remove(t)
            self.metadata._remove_table(t.key, t.schema)

        #now we need to return an instance of the deleted database
        return self.serialize_sql_object()

    def DatabaseDump(self, tables=None):
        return self.export_to_file({
            table.key: self.wl_transpose(
                (force_text(c.key) for c in table.columns),
                self.connection.execute(sa.select([table])))
            for table in to_table(self.metadata, tables)
        })

    def WithEnvironment(self, expr, env):
        for key, value in self.parse_env(**env):
            setattr(self, key, value)
        return self.evaluate(expr)

    def DatabaseInspect(self, tables=None, include_related_tables = False):
        #this API can just use to_metadata

        if tables is None:
            self.metadata.reflect(include_related_tables = include_related_tables)
        else:
            self.metadata.reflect(only=tuple(iterate(tables)), include_related_tables = include_related_tables)

        dialect = self.connection.create_engine().dialect

        for table in self.metadata.tables.values():
            for column in table.c.values():

                info = get_type_info(column.type, dialect = dialect)

                if info.name == 'Unsupported':
                    messages.inspection_cannot_infer_type.send_message(
                        repr(column.type),
                        column.name, 
                        table.name
                    )

        #we need to inspect the timezone.
        #postgresql: show timezone;
        #mysql: SELECT @@global.time_zone, @@session.time_zone, @@system_time_zone; is returning something like CET

        return self.serialize_sql_object()

    def _parse_insert_data(self, table, data):
        for instance in data:
            full = {column.key: None for column in table.c}
            for key, value in instance.items():
                full[key] = self.evaluate(value)

            yield full

    @to_dict
    def insert_data(self, data, batch_size=100):

        # TODO: close connection properly

        trans = self.connection.begin()

        tables = tuple(to_table(self.metadata, data.keys()))
        tables = tuple(
            filter(lambda t: t in tables, self.metadata.sorted_tables))

        try:
            for table in tables:

                l = 0
                for batch in partition(
                        self._parse_insert_data(table, data[table.key]),
                        batch_size):
                    self.connection.execute(table.insert(), batch)
                    l += len(batch)

                yield table.key, l

            trans.commit()
        except:
            trans.rollback()
            raise

    def DatabaseInsert(self, data):
        return self.export_to_file(self.insert_data(data))

    # API for queries

    def DatabaseQueryToString(self, query):
        self.alias = {}
        return force_text(
            render_query(
                self.evaluate(query), self.connection.create_engine()))

    def DatabaseQuery(self, query):

        #query is supposed to be a WolframLanguageExpression that needs to be evaluated

        if isinstance(query, six.string_types):
            result_proxy = self.connection.execute(query)

        else:

            #in case query is an Expression, we need to define an attribute that we are using to build the actual query
            self.alias = {}

            query = self.evaluate(query)
            result_proxy = self.connection.execute(query)

        #returning an assoc using column description
        return self.export_to_file(
            dict(
                QueryKeys=map(force_text, result_proxy.keys()),
                # we are iterating the cursor directly in order to remove all postprocessors that are happening when iterating the result proxy.
                QueryRows=result_proxy.cursor and result_proxy.cursor.fetchall() or (),
            ))

    def SAFullQuery(self, *statements):
        result = None
        for obj in statements:
            result = self.evaluate(obj)

        return result

    # aliasing, table and column selections

    def SAAlias(self, name):
        return self.alias[name]

    def SACorrelate(self, q, table):
        return self.evaluate(q).correlate(self.evaluate(table))

    def SACreateAlias(self, query, name):

        if isinstance(query, six.string_types):
            target = self.metadata.get_table_from_name(query).alias(name)
        else:
            target = self.evaluate(query).alias(name)

        self.alias[name] = target

        return target

    def SATableField(self, table, name):
        if isinstance(table, six.string_types):
            return self.metadata.get_table_from_name(
                table).columns.get_column_from_name(name)
        return self.evaluate(table).columns.get_column_from_name(name)

    def SALabel(self, field, name):
        return self.as_parameter(field).label(name)

    # select operations
    def SASelect(self, fields):
        return sa.select(map(self.as_scalar, fields))

    def SASelectFrom(self, query, alias):
        result = self.evaluate(query).select_from(
            self.evaluate(alias))
        return prevent_implicit_joins(result)

    def SAWhere(self, query, where):
        result = self.evaluate(query).where(self.evaluate(where))
        return prevent_implicit_joins(result)

    def SAGroupBy(self, query, fields):
        return prevent_implicit_joins(
            self.evaluate(query).group_by(*self.evaluate(fields))
        )

    def SADistinct(self, mode, arg, *rest):
        if mode == "query":
            return prevent_implicit_joins(
                self.evaluate(arg).distinct(*map(self.evaluate, rest))
            )
        elif mode == "aggregation":
            return sa.distinct(self.evaluate(arg))
        else:
            raise messages.unsupported_backend.as_exception(
                'DISTINCT', self.backend_name())

    def SAOffsetLimit(self, query, off, lim):
        res = self.evaluate(query)
        if off is not None:
            res = res.offset(self.evaluate(off))
        if lim is not None:
            res = res.limit(self.evaluate(lim))
        return prevent_implicit_joins(res)

    #join and groupby

    def SAJoin(self, f, s, condition=None):
        return self.evaluate(f).join(
            self.evaluate(s), self.evaluate(condition))


    def SAOuterJoin(self, f, s, jtype="Left", condition=None):

        if self.backend_name() == 'sqlite' and jtype == "Full":
            raise messages.unsupported_backend.as_exception(
                'FULL OUTER JOIN', self.backend_name())

        if jtype not in ("Left", "Full"):
            raise messages.unsupported_backend.as_exception(
                'JOIN %s' % jtype.upper(), self.backend_name())

        return self.evaluate(f).outerjoin(
            self.evaluate(s),
            onclause=self.evaluate(condition),
            full=jtype == "Full")

    # ordering

    def SAOrderBy(self, table, ordering):
        return prevent_implicit_joins(
            self.evaluate(table).order_by(*map(self.evaluate, ordering))
        )

    def SAAscending(self, field):
        return self.evaluate(field).asc()

    def SADescending(self, field):
        return self.evaluate(field).desc()

    # mathematical operations and comparison
    # they need to be translated as scalar

    def as_scalar(self, expr):
        expr = self.as_parameter(expr)
        try:
            return expr.as_scalar()
        except AttributeError:
            return expr

    def as_parameter(self, expr):
        expr = self.evaluate(expr)
        if is_data_type(expr):
            return sa.expression.literal(expr)
        return expr

    def SALess(self, f, s):
        return sa.operators.lt(self.as_scalar(f), self.as_scalar(s))

    def SAGreater(self, f, s):
        return sa.operators.gt(self.as_scalar(f), self.as_scalar(s))

    def SALessEqual(self, f, s):
        return sa.operators.le(self.as_scalar(f), self.as_scalar(s))

    def SAGreaterEqual(self, f, s):
        return sa.operators.ge(self.as_scalar(f), self.as_scalar(s))

    def SAEqual(self, f, s):
        return sa.operators.eq(self.as_scalar(f), self.as_scalar(s))

    def SAUnequal(self, f, s):
        return sa.operators.ne(self.as_scalar(f), self.as_scalar(s))

    def SAIn(self, s, iterable):

        if isinstance(iterable, (tuple, list, set, frozenset)):
            iterable = map(self.as_parameter, iterable)
        else:
            iterable = self.as_parameter(iterable)

        return self.as_parameter(s).in_(iterable)

    def SACoalesce(self, *args):
        return sa.functions.coalesce(*map(self.as_parameter, args))

    @backend_based
    def SARound(self, x):
        return sa.functions.round(self.as_scalar(x))

    def SARound_mssql(self, x):
        return sa.functions.round(self.as_scalar(x), 0)

    def SAFloor(self, x):
        return sa.functions.floor(self.as_scalar(x))

    @backend_based
    def SACeiling(self, x):
        return sa.functions.ceil(self.as_scalar(x))

    def SACeiling_mssql(self, x):
        return sa.functions.ceiling(self.as_scalar(x))

    @backend_based
    def SALog(self, x):
        return sa.functions.log(self.as_scalar(x))

    def SALog_oracle(self, x):
        return sa.functions.ln(self.as_scalar(x))

    def SALog_postgresql(self, x):
        return sa.functions.ln(self.as_scalar(x))
    

    def SASin(self, x):
        return sa.functions.sin(self.as_scalar(x))

    def SACos(self, x):
        return sa.functions.cos(self.as_scalar(x))

    def SATan(self, x):
        return sa.functions.tan(self.as_scalar(x))

    def SAArcTan(self, x):
        return sa.functions.atan(self.as_scalar(x))

    def SAArcTan2(self, y, x):
        return sa.functions.atan2(self.as_scalar(y), self.as_scalar(x))

    def SAArcSin(self, x):
        return sa.functions.asin(self.as_scalar(x))

    def SAArcCos(self, x):
        return sa.functions.acos(self.as_scalar(x))

    @backend_based
    def SABitAnd(self, *args):
        return self.SACustomOperator("&", *args)

    def SABitAnd_oracle(self, *args):
        return reduce(
            lambda a, b: sa.functions.BITAND(self.as_scalar(a), self.as_scalar(b)),
            args
        )

    @backend_based
    def SABitOr(self, *args):
        return self.SACustomOperator("|", *args)

    def SABitOr_oracle(self, *args):
        return self.SABitNot(self.SABitAnd(*map(self.SABitNot, args)))


    @backend_based
    def SABitXor(self, *args):
        return self.SACustomOperator("^", *args)

    @backend_based
    def SABitXor_postgresql(self, *args):
        return self.SACustomOperator("#", *args)

    def SABitXor_sqlite(self, *args):
        return reduce(
            lambda a, b: sa.functions.xor(self.as_scalar(a), self.as_scalar(b)),
            args
        )

    def SABitXor_oracle(self, *args):
        return reduce(
            lambda a, b: self.SABitAnd(
                self.SABitNot(self.SABitAnd(self.as_scalar(a), self.as_scalar(b))), 
                self.SABitNot(self.SABitAnd(self.SABitNot(self.as_scalar(a)), self.SABitNot(self.as_scalar(b))))
            ),
            args
        )

    @backend_based
    def SABitNot(self, x):
        return self.SACustomUnary('~', x)

    def SABitNot_mysql(self, x):
        return self.SASubtract(self.SAMultiply(x, -1), 1)

    def SABitNot_oracle(self, x):
        return self.SASubtract(self.SAMultiply(x, -1), 1)

    @backend_based
    def SABitShiftLeft(self, x, n):
        return self.SACustomOperator('<<', x, n)

    def SABitShiftLeft_oracle(self, x, n):
        return self.SAMultiply(x, self.SAPower(2, n))

    def SABitShiftLeft_mssql(self, x, n):
        return self.SAMultiply(x, self.SAPower(2, n))

    @backend_based
    def SABitShiftRight(self, x, n):
        return self.SACustomOperator('>>', x, n)

    def SABitShiftRight_oracle(self, x, n):
        return self.SAFloor(self.SADivide(x, self.SAPower(2, n)))

    def SABitShiftRight_mssql(self, x, n):
        return self.SADivide(x, self.SAPower(2, n))

    @backend_based
    def SAPower(self, f, s):
        # TODO: power keyword is supported by postgresql, mysql, oracle and sqlserver
        # However, we'll need to check that all other backends support it

        return sa.functions.power(self.as_scalar(f), self.as_scalar(s))

    def SAPower_sqlite(self, f, s):

        #POWER is not supported by SQLITE
        #however there is a C extension that we can use:
        #https://www.sqlite.org/contrib?orderby=date
        #https://www.sqlite.org/contrib/download/extension-functions.c?get=25

        if six.JYTHON:
            raise messages.unsupported_backend.as_exception(
                'POWER', self.backend_name())
        return sa.functions.power(
            self.as_scalar(f), self.as_scalar(s))

    @backend_based
    def SAMod(self, f, s):
        return sa.operators.mod(self.as_scalar(f), self.as_scalar(s))

    def SAMod_oracle(self, f, s):
        return sa.functions.MOD(self.as_scalar(f), self.as_scalar(s))

    def SAAbs(self, n):
        return sa.functions.abs(self.as_scalar(n))
    
    def SAMinus(self, f):
        return sa.operators.mul(self.as_scalar(f), -1)

    @backend_based
    def SABetween(self, a, b, c):
        return sa.expression.between(
            self.as_scalar(a), self.as_scalar(b), self.as_scalar(c))

    def SABetween_sqlite(self, a, b, c):
        return sa.cast(
            sa.expression.between(
                self.as_scalar(a), self.as_scalar(b), self.as_scalar(c)),
            types.wolfram.Boolean())

    def SAAdd(self, *args):
        return reduce(sa.operators.add, map(self.as_scalar, args))

    def SAMultiply(self, *args):
        return reduce(sa.operators.mul, map(self.as_scalar, args))

    def SASubtract(self, f, s):
        return sa.operators.sub(self.as_scalar(f), self.as_scalar(s))

    def SADivide(self, f, s):
        return sa.operators.div(self.as_scalar(f), self.as_scalar(s))

    @backend_based
    def SAQuotient(self, f, s):
        return self.SADivide(f, s)

    def SAQuotient_oracle(self, f, s):
        return sa.functions.FLOOR(self.SADivide(f, s))

    def SAQuotient_mysql(self, f, s):
        return sa.functions.FLOOR(self.SADivide(f, s))

    @backend_based
    def SAConcat(self, *args):
        return sa.functions.concat(*map(self.as_parameter, args))

    def SAConcat_mssql(self, *args):
        return self.SAAdd(*args)

    def SAConcat_oracle(self, *args):
        if len(args) == 1:
            return self.as_parameter(first(args))
        return reduce(lambda a, b: sa.functions.concat(a, b), map(self.as_parameter, args))

    def SAConcat_sqlite(self, *args):
        return self.SACustomOperator("||", *args)

    # Aggregations

    def SAStandardDeviation(self, aggregation):
        return sa.functions.stddev(self.as_parameter(aggregation))

    def SAVariance(self, aggregation):
        return sa.functions.variance(self.as_parameter(aggregation))

    def SASum(self, aggregation):
        return sa.functions.sum(self.as_parameter(aggregation))

    def SACount(self, aggregation):
        return sa.functions.count(self.as_parameter(aggregation))

    def SACountAll(self):
        return sa.functions.count()

    def SAMax(self, aggregation):
        return sa.functions.max(self.as_parameter(aggregation))

    def SAMin(self, aggregation):
        return sa.functions.min(self.as_parameter(aggregation))

    @backend_based
    def SALeast(self, *args):
        return sa.functions.least(*map(self.as_scalar, args))

    def SALeast_sqlite(self, *args):
        return sa.functions.min(*map(self.as_scalar, args))

    def SALeast_mssql(self, *args):
        if len(args) <= 1:
            return self.as_scalar(first(args, None))
        return reduce(
            lambda a, b: sa.expression.case(((sa.operators.lt(a, b), a), ), else_ = b), 
            map(self.as_scalar, args)
        )

    @backend_based
    def SAGreatest(self, *args):
        return sa.functions.greatest(*map(self.as_scalar, args))

    def SAGreatest_sqlite(self, *args):
        return sa.functions.max(*map(self.as_scalar, args))

    def SAGreatest_mssql(self, *args):
        if len(args) <= 1:
            return self.as_scalar(first(args, None))
        return reduce(
            lambda a, b: sa.expression.case(((sa.operators.gt(a, b), a), ), else_ = b), 
            map(self.as_scalar, args)
        )

    def SAMean(self, aggregation):
        return sa.functions.avg(self.as_parameter(aggregation))

    @unsupported("Any", "sqlite")
    def SAAny(self, subquery):
        return sa.expression.any_(self.evaluate(subquery))

    @unsupported("All", "sqlite")
    def SAAll(self, subquery):
        return sa.expression.all_(self.evaluate(subquery))

    # logic operations and comparison

    def SAOr(self, *conditions):
        return sa.expression.or_(*map(self.as_parameter, conditions))

    def SAAnd(self, *conditions):
        return sa.expression.and_(*map(self.as_parameter, conditions))

    def SANot(self, condition):
        return sa.expression.not_(self.as_parameter(condition))

    def SAExists(self, condition):
        return sa.expression.exists(self.as_parameter(condition))

    def SACase(self, conditions, values):

        zipped = list(zip(conditions, values))

        if zipped[-1][0] is True:
            else_ = self.as_parameter(zipped.pop(-1)[-1])
        else:
            else_ = None

        return sa.expression.case(
            ((self.as_parameter(condition), self.as_parameter(value))
             for condition, value in zipped),
            else_=else_)
            
    def SAByteArray(self, expr):
        return self.evaluate(expr) or b''  

    #String manipulations

    @backend_based
    def SASubstr(self, s, i, l):
        return sa.functions.substr(self.as_parameter(s), self.as_parameter(i), self.as_parameter(l))

    def SASubstr_mssql(self, s, i, l):
        return sa.functions.substring(self.as_parameter(s), self.as_parameter(i), self.as_parameter(l))

    def SAStringLength(self, s):
        return sa.functions.length(self.as_parameter(s))

    @backend_based
    def SARegexp(self, f, s, case_insensitive=False):
        raise NotImplementedError

    def SARegexp_sqlite(self, f, s, case_insensitive=False):
        return sa.functions.INTERNAL_REGEXP(
            self.as_parameter(f), 
            self.as_parameter(s),
            bool(case_insensitive)
        )

    def SARegexp_postgresql(self, f, s, case_insensitive=False):
        return self.SACustomOperator(case_insensitive and '~*' or '~', f, s)

    def SARegexp_mysql(self, f, s, case_insensitive=False):
        #note that this function cannot be used on mysql 5.7.
        #for now we are not checking the support, the REGEXP operator cannot be used because it's always doing a case insensitive match
        return sa.cast(
            #we need to add multiline and casesens / insensitive flags
            sa.functions.REGEXP_LIKE(
                self.as_parameter(f), self.as_parameter(s),
                case_insensitive and 'im' or 'cm'),
            types.wolfram.Boolean())

    def SARegexp_oracle(self, f, s, case_insensitive=False):
        return sa.functions.REGEXP_LIKE(
            self.as_parameter(f), 
            self.as_parameter(s),
            case_insensitive and 'i' or 'c'
        )

    def SARegexp_mssql(self, *args, **opts):
        raise messages.unsupported_backend.as_exception(
            'REGEXP', self.backend_name())

    def SACustomUnary(self, operator_name, x):
        return sa.expression.UnaryExpression(
            self.as_scalar(x),
            sa.operators.custom_op(operator_name)
        )

    def SACustomOperator(self, operator_name, *args):

        if len(args) == 1:
            return self.as_parameter(first(args))

        return reduce(
            lambda a, b, operator_name = operator_name: sa.expression.BinaryExpression(
                self.as_parameter(a), self.as_parameter(b),
                sa.operators.custom_op(operator_name)
            ), 
            args
        )

    # Type conversions

    def SACast(self, expr, database_type):
        return sa.cast(
            self.as_scalar(self.evaluate(expr)),
            self.evaluate(database_type),
            dialect=self.connection.create_engine().dialect)

    # date literals

    def SADateTime(self,
                   timezone,
                   year=1970,
                   month=1,
                   day=1,
                   hour=0,
                   minute=0,
                   second=0):
        d = datetime.datetime(
            year=int(year),
            month=int(month),
            day=int(day),
            hour=int(hour),
            minute=int(minute),
            second=int(math.floor(second)),
            microsecond=int((float(second) - math.floor(second)) * 1000000),
        )

        return sa.Compilable(new_datetime(d, d, tzinfo=timezone))

    def SATime(self, timezone, hour=0, minute=0, second=0):
        return sa.Compilable(
            new_time(
                datetime.time(
                    hour=int(hour),
                    minute=int(minute),
                    second=int(math.floor(second)),
                    microsecond=int(
                        (float(second) - math.floor(second)) * 1000000),
                ),
                tzinfo=timezone))

    def SADate(self, timezone = None, year = 1970, month=1, day=1):
        #timezone is dropped, mathematica dates are still storing this info but it makes no sense in a date
        return sa.Compilable(
            datetime.date(
                year=int(year),
                month=int(month),
                day=int(day),
            ))

    def SAIsNull(self, value):
        return self.SACustomOperator('IS', value, sa.expression.text('NULL'))

    @backend_based
    def SABoolean(self, value):
        #timezone is dropped, mathematica dates are still storing this info but it makes no sense in a date
        if value:
            return sa.expression.True_()
        return sa.expression.False_()

    def SABoolean_oracle(self, value):
        return self.SAEqual(int(value), 1)

    def SABoolean_mssql(self, value):
        return self.SAEqual(int(value), 1)

    # date operations
    @backend_based
    def SADateAdd(self, date, seconds):
        return self.SAAdd(date, seconds)

    def SADateAdd_mysql(self, date, seconds):
        return sa.functions.TIMESTAMPADD(
            sa.expression.text('second'),
            self.as_parameter(seconds),
            self.as_parameter(date),
        )

    def SADateAdd_mssql(self, date, seconds):
        return sa.functions.DATEADD(
            sa.expression.text('second'),
            self.as_parameter(seconds),
            self.as_parameter(date),
        )

    @backend_based
    def SADateDifference(self, date1, date2):
        return self.SASubtract(date1, date2)

    def SADateDifference_mssql(self, date1, date2):
        return sa.functions.DATEDIFF(
            sa.expression.text('second'), self.as_parameter(date1),
            self.as_parameter(date2))

    @backend_based
    def SASecondsToTimeQuantity(self, seconds):
        return self.as_parameter(seconds)

    def SASecondsToTimeQuantity_postgresql(self, seconds):
        return self.SAMultiply(
            self.as_parameter(seconds), sa.cast('1 seconds', "TimeQuantity"))

    def SASecondsToTimeQuantity_oracle(self, seconds):
        return sa.functions.NUMTODSINTERVAL(self.as_parameter(seconds), 'second')

    @backend_based
    def SAUnixTime(self, arg):
        #postgres implementation
        return NotImplementedError(arg)

    def SAUnixTime_postgresql(self, arg):
        return sa.functions.extract('epoch', self.as_parameter(arg))

    def SAUnixTime_mysql(self, arg):
        return sa.functions.UNIX_TIMESTAMP(self.as_parameter(arg))

    def SAUnixTime_mssql(self, arg):
        return sa.cast(
            sa.functions.DATEDIFF(
                sa.expression.text('second'), '1970-01-01',
                self.as_parameter(arg)), "Real")

    def SAUnixTime_oracle(self, arg):

        #"SELECT CAST(person.datetime AS DATE) - CAST(to_date('1-1-1970 00:00:00','MM-DD-YYYY HH24:Mi:SS') AS DATE) FROM person"

        date = sa.functions.CAST(
            self.SACustomOperator(
                'as',
                self.as_parameter(arg),
                sa.expression.text('DATE')
            ),
        )
        epoch = sa.functions.CAST(
            self.SACustomOperator(
                'as',
                self.SADate(),
                sa.expression.text('DATE')
            ),
        )

        return (date - epoch) * 86400.

    def SAUnixTime_sqlite(self, arg):
        return self.as_parameter(arg)

    @backend_based
    def SAFromUnixTime(self, arg):
        return self.SADate() + self.SASecondsToTimeQuantity(arg)

    def SAFromUnixTime_postgresql(self, arg):
        #postgres implementation
        return sa.functions.to_timestamp(self.as_parameter(arg))

    def SAFromUnixTime_mysql(self, arg):
        return sa.functions.FROM_UNIXTIME(self.as_parameter(arg))

    @backend_based
    def SANow(self):
        return sa.functions.now()

    def SANow_sqlite(self):
        return sa.cast(sa.functions.strftime('%s', 'now'), "Real")

    @backend_based
    def SAToday(self):
        return sa.functions.current_date()

    def SAToday_sqlite(self):
        return sa.cast(
            sa.functions.strftime('%s', sa.functions.date('now')), "Real")
