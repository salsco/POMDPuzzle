import os
import setuptools

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setuptools.setup(
    name = "POMDPuzzle",
    version = "0.0.1",
    author = "Salvatore Scozzari",
    description = ("A custom library to solve Partially Observable Markov Decision Processes (POMDPs)"),
    packages=setuptools.find_packages('.'),
    package_dir={
        'POMDPuzzle': 'pomdpuzzle',
    },
    setup_requires=['wheel'],
    long_description=read('README.md')
    )
