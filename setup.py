#!/usr/bin/env python3

from setuptools import find_packages, setup
from pathlib import Path

setup(
	name="MNIST Classification",
	packages=find_packages(include=["numpy","pandas","matplotlib"]),
	version="0.0.0",
	description="Classifying Numbers from the MNIST dataset using only Python and minimal Libraries",
	author="Mark Tobin",
	author_email='04x0xx@gmail.com',
	license="GPL-3.0",
	install_requires=[],
	setup_requires=[],
	long_description = ( Path(__file__).parent / "README.md" ).read_text(),
	long_description_content_type='text/markdown'
)