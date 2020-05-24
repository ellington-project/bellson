from setuptools import setup, find_packages
import argparse

with open("README.md", "r") as rh:
    long_description = rh.read()

setup(name='bellson',
      version='0.0.1',
      author="Adam Harries",
      author_email="harries.adam@gmail.com",
      description="ML based tempo detection for swing music",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://www.github.com/ellington-project/bellson",
      packages=find_packages("."),
      include_package_data=True,
      install_requires=[
          'keras',
          'tensorflow',
          'tensorboard',
          'numpy',
          'librosa',
          'objgraph',
          'matplotlib',
          'pandas',
          'seaborn',
          'h5py',
          'importlib-resources'],
      zip_safe=False,
      entry_points={
          "console_scripts": [
              "bellson-preprep = bellson.apps.util.preprep:entrypoint",
              "bellson-train = bellson.apps.tf.train:entrypoint",
              "bellson-infer = bellson.apps.tf.infer:entrypoint"
          ]
      })
