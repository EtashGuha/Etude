# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import logging

logger = logging.getLogger('wolfram.databases')


class MessageAccumulatorHandler(logging.Handler):
    def __init__(self):
        self.records = []
        super(MessageAccumulatorHandler, self).__init__()

    def emit(self, record):
        self.records.append(record)

    def __iter__(self):
        for record in self.records:
            yield record.msg

    def register_logger(self, logger=logger):
        logger.addHandler(self)

    def remove_logger(self, logger=logger):
        logger.removeHandler(self)
        self.close()


def send_message(payload):
    return logger.warning(payload)


def new_accumulator(**opts):
    accumulator = MessageAccumulatorHandler()
    accumulator.register_logger(**opts)
    return accumulator
