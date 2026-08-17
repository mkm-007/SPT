from setuptools import setup, find_packages

setup(
    name='neurein',
    version='0.1.0',
    description='A deep learning framework built from scratch',
    packages=find_packages(),
    install_requires=['numpy'],
    python_requires='>=3.8',
)
