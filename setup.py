import os
from setuptools import setup

# from numpy.distutils.core import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="dgp-aep-mcm",
    version="0.1",
    author="Gonzalo Hernandez-Muñoz, Daniel Hernandez-Lobato",
    author_email="gonzalo.hernandez@uam.es, daniel.hernandez@uam.es",
    license="Apache License 2.0",
    description=(
        "A package for Deep GPs in python using approximate expectation propagation and Monte Carlo Methods."
    ),
    keywords="Deep GPs",
    packages=["dgp_aepmcm", "dgp_aepmcm.kernel", "dgp_aepmcm.layers", "dgp_aepmcm.nodes"],
    long_description=read("README.md"),
    install_requires=[
        "numpy",
        "tensorflow>=1.12.0",
        "scipy",
    ],  # Requires 1.12.0 but not available in pip. conda install -c conda-forge tensorflow=1.12.0
)
