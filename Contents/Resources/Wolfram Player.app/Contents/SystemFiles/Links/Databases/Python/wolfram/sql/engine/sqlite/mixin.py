# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals


class SQLiteMixin:

    #this needs to be set to an empty dict so that names like DATETIME and friends are not automatically converted to internal types, but the real type affinity is used.
    ischema_names = {}
