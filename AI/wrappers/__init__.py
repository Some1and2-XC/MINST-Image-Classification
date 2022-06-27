#!/usr/bin/env python3

from os import path, chdir
from functools import wraps

base_path = path.abspath("")

def b(func):
	# Wraps a function to move to the Root Folder After Function Completion
	@wraps(func)
	def wrapper(*args, **kwargs):
		global base_path
		r = func(*args, **kwargs)
		chdir(base_path)
		return r
	return wrapper
