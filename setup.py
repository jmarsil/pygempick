from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='pygempick', 
      version='1.1', 
      description='Open Source Batch Gold Particle Picking & Procesing for Immunogold Diagnostics',
      long_description = readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'Operating System :: MacOS',
        'Operating System :: Unix',
        'Topic :: Education',
        'Topic :: Utilities',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Mathematics',
      ],
      keywords='immunogold batch gold particle picker EM micrograph processing correlation \
      analysis IGEM diagnostics ',
      url='https://github.com/jmarsil/pygempick',
      author='Joseph Marsilla',
      author_email='joseph.marsilla@mail.utoronto.ca',
      license='MIT',
      packages=['pygempick'],
      install_requires=['numpy','pandas', 'matplotlib'],
      include_package_data=True,
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],)
