
from setuptools import setup, find_packages

setup(
    name="hypera1_training",
    version="0.1",
    packages=find_packages(),
    py_modules=["package"],
    entry_points={
        'console_scripts': [
            'hypera1_training=package:main',
        ],
    },
)
