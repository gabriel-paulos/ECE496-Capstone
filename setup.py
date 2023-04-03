from setuptools import setup

setup(
name="capstone-proj",
version="1.0",
author = "Capstone Group 27",
license="MIT",
py_modules=['src/cli/main'],
install_requires=["torch>=1.13.1",
"torchaudio>=0.13.1",
"tensorflow>=2.11.0",
"transformers>=4.26.1",
"click>=8.1.3",
"moviepy>=1.0.3"],
include_package_data=True,
package_dir = {'':'src'},
)
