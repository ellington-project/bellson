'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

setup(name='bellson',
      version='1.0',
      packages=find_packages(),
      include_package_data=True,
      description='Bellson BPM keras model on Cloud ML Engine',
      author='Adam Harries',
      author_email='harries.adam@gmail.com',
      license='GPL-3.0',
      install_requires=[
          'keras',
          'tensorflow',
          'numpy',
          'librosa',
          'objgraph', 
          'matplotlib',
          'h5py'],
      zip_safe=False)