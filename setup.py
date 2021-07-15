import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="CLIPDetection",
    py_modules=["CLIPDetection"],
    version="1.0",
    packages=find_packages(),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    package_data={'CLIPDetection': ['clip/bpe_simple_vocab_16e6.txt.gz']},
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)
