# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals


class MSMixin(object):

    def get_unique_constraints(self, *args, **opts):
        #this is not implemented for mssql
        #this methods prevents sqlalchemy from raising NotImplementedError
        return ()