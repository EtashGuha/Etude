# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import os
import tempfile
import platform

from wolfram.cli.dispatch import SimpleCommand
from wolfram.utils.system import collect_python_files
from wolframclient.utils.decorators import decorate
from wolframclient.utils.functional import first, iterate
from wolframclient.utils.importutils import import_string, module_path

DEPENDENCIES = [
    'decimal',
    'psycopg2',
    'pymssql',
    'pymysql',
    'pytz',
    'sqlalchemy',
]

EXPANDED_DEPENDENCIES = [
    ('wolframclient', lambda name: not name.startswith('wolframclient.tests')),
    ('wolfram',       lambda name: not name.startswith('wolfram.tests')),
]

EXCLUDED = [
    'setuptools', 'pyinstaller', 'distutils', 'pip', 'numpy', 'cx_Oracle'
]
    

def collect_module_tree(module):
    root = module_path(module)
    for fullpath in collect_python_files(root):
        yield module_path_from_filename(root, fullpath)


def module_path_from_filename(root, fullpath):
    rel = os.path.relpath(fullpath, os.path.join(root, os.path.pardir))
    rel = first(os.path.splitext(rel))
    rel = rel.split(os.sep)
    if rel[-1] == '__init__':
        return ".".join(rel[:-1])
    return ".".join(rel)


class Command(SimpleCommand):

    #def distribution_name(self):
    #    return 'distribution-%s-%s-%s' % (
    #        platform.system() == 'Darwin' and 'MacOSX' or platform.system(),
    #        platform.uname()[4][:3],
    #        sys.maxsize > 2**32 and 64 or 32
    #    )

    def add_arguments(self, parser):
        parser.add_argument('--folder', dest='folder', default=None)
        parser.add_argument('--name', dest='name', default=None)
        parser.add_argument(
            '--path', dest='paths', action='append', default=None)
        parser.add_argument('--specpath', dest='specpath', default=None)

    @decorate(frozenset)
    def all_dependencies(self):
        for dep in DEPENDENCIES:
            yield dep
        for dep, test in EXPANDED_DEPENDENCIES:
            yield dep

    @decorate(frozenset)
    def exploded_dependencies(self):
        for module in self.all_dependencies():
            yield module

        for module, test in EXPANDED_DEPENDENCIES:
            yield module
            for sub in filter(test, collect_module_tree(module)):
                yield sub

    @decorate(frozenset)
    def import_paths(self):
        for module in self.all_dependencies():
            yield module_path(module, os.pardir)

    def run_arguments(self, name=None, folder=None, paths=None, specpath=None):
        yield module_path('wolfram', 'distribution.py')

        yield '--distpath'
        yield folder and os.path.abspath(folder) or module_path(
            'wolfram', os.pardir)

        yield '--name'
        yield name or 'distribution'

        for path in iterate(self.import_paths(), paths or ()):
            yield '--paths'
            yield os.path.abspath(path)
            
        if specpath:
            if os.path.abspath(specpath):
                print("Using the path %s as a specpath" % specpath)
                yield '--specpath'
                yield os.path.abspath(specpath)
            else:
                print("Passed invalid option for specpath: %s. Will use the default." % specpath)

        for module in self.exploded_dependencies():
            yield '--hiddenimport'
            yield module

        for module in EXCLUDED:
            yield '--exclude-module'
            yield module

        yield '--specpath'
        yield tempfile.gettempdir()

        yield '--onefile'
        yield '--noconfirm'

    def handle(self, folder=None, name=None, paths=(), specpath=None, **opts):

        args = tuple(self.run_arguments(
            name=name, folder=folder, paths=paths, specpath=specpath
        ))

        print(" ".join(args))

        os.chdir(tempfile.gettempdir())

        import_string('PyInstaller.__main__.run')(args)
