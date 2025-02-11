# Minimal setup that uses requirements.txt
from setuptools import setup, find_packages

setup(
    name='compositional-format-benchmarking',
    version='0.1',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=open('requirements.txt').read().splitlines()
)