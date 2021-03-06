#!/usr/bin/env python3

from setuptools import find_packages, setup
from pathlib import Path

setup(
	name="AI",
	packages=find_packages(),
	include_package_data = True,
	version="0.0.0",
	description="Classifying Numbers from the MNIST dataset using only Python and minimal Libraries",
	author="@Some1and2",
	author_email='04x0xx@gmail.com',
	license="GPL-3.0",
	install_requires=["numpy","pandas","matplotlib", "idx2numpy==1.2.3"],
	setup_requires=[],
	long_description = ( Path(__file__).parent / "README.md" ).read_text(),
	long_description_content_type='text/markdown'
)