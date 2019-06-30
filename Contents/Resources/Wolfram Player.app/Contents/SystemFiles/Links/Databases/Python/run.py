# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

if __name__ == '__main__':

    #this will perform an auto install of missing modules using PIP
    #this won't be used in production, but it's handy when we are ginving this paclet to other developers

    try:
        import wolframclient
    except ImportError:
        import sys
        import os

        #this will assume that WolframClientForPython is in the same folder of this one
        #used for development only

        path = os.path.normpath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                os.pardir,
                os.pardir,
                os.pardir,
                'WolframClientForPython'
            )
        )

        if os.path.exists(path) and not path in sys.path:
            sys.path.insert(0, path)

        try:
            import wolframclient
        except ImportError:
            raise ImportError(
                'wolframclient is not available. it needs to be on sys.path"'
            )

    from wolfram.cli.dispatch import execute_from_command_line

    execute_from_command_line(distribution = False)