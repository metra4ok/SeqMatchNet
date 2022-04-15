import os
from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='seqmatchnet',
      version='1.0',
      description='SeqMatchNet model code',
      author='Sourav Garg',
      packages=['seqmatchnet'],
     )
