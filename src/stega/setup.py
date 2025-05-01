import os
from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize(os.path.join('stega', 'stega_cy.pyx'),
                            annotate=False,
                            compiler_directives={
                                'boundscheck': False,
                                'wraparound': False,
                                'language_level': 3
                            }))

setup(ext_modules=cythonize(os.path.join('stega', 'random_sample_cy.pyx'),
                            annotate=False,
                            compiler_directives={
                                'boundscheck': False,
                                'wraparound': False,
                                'language_level': 3
                            }))