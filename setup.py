from setuptools import setup, find_packages
import os
this_directory = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='quantreo',  # The name of the package as it will appear on PyPI.
    version='0.0.5',
    description='Python library for quantitative trading',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Lucas Inglese',
    author_email='lucas@quantreo.com',
    # url='https://github.com/your_username/my_quant_lib',
    packages=['quantreo', 'quantreo.features_engineering'],
    install_requires=["numpy", "pandas", "numba"],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License'],
)