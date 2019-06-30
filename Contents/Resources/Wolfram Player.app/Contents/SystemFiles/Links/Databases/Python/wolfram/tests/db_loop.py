# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import os

from wolfram.cli.commands.listener_loop import (BYTES_LEN, APPEND_MODE, WRITE_MODE, listener_loop,
                                                write_bytes)
from wolframclient.language import wl
from wolframclient.serializers import export
from wolframclient.utils.encoding import force_bytes, force_text
from wolframclient.utils.tests import TestCase as BaseTestCase


class TestCase(BaseTestCase):
    def compare(self, string_version, result):
        self.assertEqual(string_version, export(result, target_format='wxf'))

    def test_listener_loop(self):

        messages = [wl.Null, 2]

        r, w = os.pipe()

        with os.fdopen(w, 'w') as stream:

            for expr in messages:

                message = export(expr, target_format='wxf')

                pre, post = message[0:2], message[2:]

                write_bytes(stream, APPEND_MODE)
                write_bytes(
                    stream,
                    force_bytes(force_text(len(pre)).zfill(BYTES_LEN)))
                write_bytes(stream, pre)

                write_bytes(stream, WRITE_MODE)
                write_bytes(
                    stream,
                    force_bytes(force_text(len(post)).zfill(BYTES_LEN)))
                write_bytes(stream, post)

        listener_loop(
            message_limit=len(messages),
            remove_stdout=False,
            stdin=os.fdopen(r, 'r'))
