# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import logging
import os
import sys
import warnings
from itertools import repeat

from wolfram.cli.dispatch import SimpleCommand
from wolfram.execution.evaluate import evaluate
from wolfram.settings import settings
from wolfram.utils.autoreload import autoreload
from wolframclient.language import wl
from wolframclient.language.side_effects import side_effect_logger
from wolframclient.serializers import export
from wolframclient.utils import six
from wolframclient.utils.encoding import force_bytes, force_text

BYTES_LEN = 12
MSG_RECEIVED = b'K'
APPEND_MODE = b'A'
WRITE_MODE = b'W'


class SideEffectSender(logging.Handler):
    def emit(self, record):
        if isinstance(sys.stdout, StdoutProxy):
            sys.stdout.send_side_effect(record.msg)


side_effect_logger.addHandler(SideEffectSender())

if six.PY2:

    def write_bytes(stream, bytes, flush=False):
        stream.write(bytes)
        if flush:
            stream.flush()

    def read_bytes(stream, bytes, flush=False):
        return stream.read(bytes)

else:

    def write_bytes(stream, bytes, flush=False):
        stream.buffer.write(bytes)
        if flush:
            stream.flush()

    def read_bytes(stream, bytes, flush=False):
        return stream.buffer.read(bytes)

def write_msg(stream, bytes, flush=True):
    write_bytes(stream, force_bytes(force_text(len(bytes)).zfill(BYTES_LEN)))
    write_bytes(stream, bytes, flush=flush)


def write_msg_received(stream, flush=True):
    write_bytes(stream, MSG_RECEIVED, flush=flush)


class StdoutProxy:
    def __init__(self, stream):
        self.stream = stream
        self.clear()

    def clear(self):
        self.current_line = []
        self.lines = []

    def write(self, message):
        messages = force_text(message).split("\n")

        if len(messages) == 1:
            self.current_line.extend(messages)
        else:
            self.current_line.append(messages.pop(0))
            rest = messages.pop(-1)

            self.lines.extend(messages)
            self.flush()
            if rest:
                self.current_line.append(rest)

    def flush(self):
        if self.current_line or self.lines:
            self.send_lines(''.join(self.current_line), *self.lines)
            self.clear()

    def send_lines(self, *lines):
        if len(lines) == 1:
            return self.send_side_effect(wl.Print(*lines))
        elif lines:
            return self.send_side_effect(
                wl.CompoundExpression(*map(wl.Print, lines)))

    def send_side_effect(self, expr):

        write_msg(
            self.stream,
            export(
                wl.Databases.Python.DBKeepListening(expr),
                target_format='wxf',
                target_kernel_version=settings.TARGET_KERNEL_VERSION))


def read_messages(stdin, stdout, message_limit=None):
    for _ in message_limit and repeat(None, message_limit) or repeat(None):
        bag  = []
        mode = APPEND_MODE
        while mode == APPEND_MODE:

            mode = read_bytes(stdin, 1)
            to_read = int(read_bytes(stdin, BYTES_LEN))
            bag.append(read_bytes(stdin, to_read))

            if mode == APPEND_MODE:
                write_msg_received(stdout)

        yield b''.join(bag)


def listener_loop(stdout=sys.stdout,
                  stdin=sys.stdin,
                  remove_stdout=True,
                  message_limit=None):

    if remove_stdout:
        sys.stdout = StdoutProxy(stdout)
        warnings.simplefilter("ignore")

    for message in read_messages(stdin, stdout, message_limit=message_limit):
        write_msg(stdout, evaluate(message))
        del message

    if remove_stdout:
        sys.stdout = stdout


class Command(SimpleCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            '--autoreload',
            dest='use_autoreload',
            default=False,
            action='store_true')

        parser.add_argument(
            '--debug', dest='debug_mode', default=False, action='store_true')

        parser.add_argument(
            '--kernel-version',
            dest='kernel_version',
            default=None,
            type=float)

        parser.add_argument(
            '--extrapath', dest='extrapath', action='append', default=None)

    def handle(self,
               use_autoreload=False,
               debug_mode=False,
               kernel_version=None,
               extrapath=None,
               **opts):

        settings.DEBUG = debug_mode
        settings.TARGET_KERNEL_VERSION = kernel_version

        for path in extrapath or ():
            sys.path.insert(0, os.path.abspath(path))

        if use_autoreload:
            autoreload(listener_loop, modules=('wolfram', 'wolframclient'))
        else:
            listener_loop()
