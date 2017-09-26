#!/usr/bin/env python

import setuptools

setuptools.setup(
    name='tfprism',
    version='0.1',
    description='Transforms your tensorflow graph to automatically do data parallelism for training',
    author='Egil Moeller',
    author_email='egil@innovationgarage.no',
    url='https://github.com/innovationgarage/tfprism',
    packages=setuptools.find_packages(),
    install_requires=[
        'tensorflow'
    ],
    extras_require={
        'server':  [
            'dnspython',
            'netifaces',
            'click',
            'pieshell'
        ]
    },
    include_package_data=True,
    entry_points='''
    [console_scripts]
    tfprism = tfprism.trainingserver:main [server]
    '''
  )
