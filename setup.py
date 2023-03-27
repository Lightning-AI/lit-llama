from setuptools import setup, find_packages

setup(
    name='lit-llama',
    version='0.1.0',
    description='Implementation of the LLaMA language model',
    author='Lightning AI',
    author_email='will@lightning.ai',
    url='https://github.com/lightning-AI/lit-llama',
    # install_requires=['lightning'],
    packages=find_packages(),
)
