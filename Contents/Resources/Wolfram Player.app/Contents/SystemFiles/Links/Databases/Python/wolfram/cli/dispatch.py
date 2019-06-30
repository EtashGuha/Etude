# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from wolframclient.cli.dispatch import DispatchCommand as DispatchCommandBase
from wolframclient.cli.utils import SimpleCommand as SimpleCommandBase
from wolframclient.utils.require import require


class DispatchCommand(DispatchCommandBase):

    modules = [] + DispatchCommandBase.modules + ['wolfram.cli.commands']
    dependencies = [
        ("psycopg2", '2.7.4'),
        ('sqlalchemy', '1.2.14'),
        ('pytz', '2018.6'),
        ('pyinstaller', '3.4'),
        ('pymysql', '0.9.2'),
        ('pymssql', '2.1.4'),
        ('cx-oracle', '7.0.0')
    ]

    #for some reason we need to manually install cython before everything else is installed
    #to do that we manually ensure cython is installed when running the cli, then the dependencies attr of the class will automatically install the other dependencies in a separated pip call
    @require(('cython', '0.28.5'))
    def main(self, *args, **opts):
        return super(DispatchCommand, self).main(*args, **opts)


class DistributionCommand(DispatchCommandBase):

    dependencies = []
    default_command = 'listener_test'

    def subcommands(self):
        #the distribution cannot discover source code location, we need to manually map the imports we want in the dist
        return {
            'listener_loop': 'wolfram.cli.commands.listener_loop.Command',
            'listener_test': 'wolfram.cli.commands.listener_test.Command',
        }


class SimpleCommand(SimpleCommandBase):
    pass
    

def patch_pip():
    
    #alternative fix to be tested
    #import tempfile
    #os.environ['PIP_REQ_TRACKER'] = tempfile.gettempdir()

    # Applying the patch mentioned here: https://github.com/pypa/pip/issues/5790
    # The patch is from: https://github.com/pypa/pip/compare/master...benoit-pierre:req_tracker_cleanup
    
    import os
    try:    
        from pip._internal.req.req_tracker import RequirementTracker, logger
        
        def monkeypatch_method(cls):
            def decorator(func):
                setattr(cls, func.__name__, func)
                return func
            return decorator   
    
        @monkeypatch_method(RequirementTracker)
        def cleanup(self):
            for req in set(self._entries):
                self.remove(req)
            remove = self._temp_dir is not None
            if remove:
                self._temp_dir.cleanup()
                del os.environ['PIP_REQ_TRACKER']
            logger.debug('%s build tracker %r',
                         'Removed' if remove else 'Cleaned', self._root) 
    except ImportError:
        pass     

def execute_from_command_line(argv=None, distribution=False):
    
    if not distribution:
        patch_pip()
    
    return (distribution and DistributionCommand
            or DispatchCommand)(argv).main()
