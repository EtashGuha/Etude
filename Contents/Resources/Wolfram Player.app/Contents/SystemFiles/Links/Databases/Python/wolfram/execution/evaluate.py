# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from wolfram.execution.evaluator import WolframLanguageEvaluator


def evaluate(payload, *args, **opts):
    with WolframLanguageEvaluator() as evaluator:
        value = evaluator.safe_wxf_evaluate_with_messages(
            payload, *args, **opts)
    return value
