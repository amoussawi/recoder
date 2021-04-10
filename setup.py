from setuptools import setup, find_packages

import recoder


setup(
  name='recsys-recoder',
  version=recoder.__version__,
  install_requires=['torch==1.8.1', 'annoy==1.17.0',
                    'numpy==1.19.5', 'scipy==1.6.2',
                    'tqdm==4.59.0', 'glog==0.3.1'],
  packages=find_packages(),
  author='Abdallah Moussawi',
  author_email='abdallah.moussawi@gmail.com',
  url='https://github.com/amoussawi/recoder',
  license='MIT'
)
