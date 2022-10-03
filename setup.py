#!/usr/scripts/env python
"""
Brkraw-bart integration
"""
from distutils.core import setup
from setuptools import find_packages
import re
import io

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open('brkbart/__init__.py', encoding='utf_8_sig').read()
    ).group(1)

__author__ = 'SungHo Lee'
__email__ = 'shlee@unc.edu'
__url__ = 'https://github.com/dvm-shlee/brkbart'

setup(name='brkbart',
      version=__version__,
      description='Bart-tool wrapping for Bruker PvDataset',
      python_requires='>3.5',
      author=__author__,
      author_email=__email__,
      url=__url__,
      license='GNLv3',
      packages=find_packages(),
      install_requires=['brkraw>=0.3.8-rc1',
                        'paralexe>=0.1.0',
                        'numpy>=1.19.0',
                        'scipy>=1.6.0',
                        'tqdm>=4.40.0',],
      entry_points={
          'console_scripts': [
              'brkbart=brkbart.scripts.brkbart:main',
          ],
      },
      classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Medical Science Apps.',
            'Natural Language :: English',
            'Programming Language :: Python :: 3'
      ],
      keywords = 'bruker bart reconstruction mri'
     )
