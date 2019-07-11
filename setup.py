from pkg_resources import DistributionNotFound, get_distribution
from setuptools import setup, find_packages

def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None


install_deps = ['tensorboardX']

if get_dist('paddlepaddle') is None and get_dist('paddlepaddle_gpu') is None:
    install_deps.append('paddlepaddle')

setup(name='atarashi',
      version='0.1',
      description='high level paddle-paddle API',
      url='https://github.com/Meiyim/paddle-estimator',
      author='Chen Xuyi',
      author_email='chen_xuyi@outlook.com',
      license='Apache 2.0',
      packages=['atarashi'],
      python_requires='>= 2.6.*',
      install_requires=install_deps)

