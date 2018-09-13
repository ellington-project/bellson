from setuptools import setup, find_packages

setup(name='bellson',
      version='0.0.1',
      author="Adam Harries", 
      author_email="harries.adam@gmail.com",
      description="ML based tempo detection for swing music",
      url = "https://www.github.com/AdamHarries/bellson",
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          'keras',
          'tensorflow-gpu',
          'tensorboard', 
          'numpy',
          'librosa',
          'objgraph', 
          'h5py'],
      zip_safe=False)