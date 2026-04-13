"""
Copyright (C) 2017  Quentin Peter

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='diffusion_device',
      version='1.0.0',
      description='Image processing on diffusion device images',
      long_description=readme(),
      long_description_content_type='text/markdown',
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
          'Programming Language :: Python :: 3 :: Only',
          'Topic :: Scientific/Engineering',
      ],
      keywords='sizing microfluidics',
      url='https://github.com/impact27/diffusion_device',
      author='Quentin Peter',
      author_email='qaep2@cam.ac.uk',
      license='GPl v3',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scipy',
          'opencv-python',
          'tifffile',
          'background_rm>=0.1.3',
          'registrator>=0.1.3',
          'natsort',
          'matplotlib',
          'pandas',
          'tk',
      ],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      include_package_data=True,
      zip_safe=False)
