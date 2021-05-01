"""Setup.py for torchflare."""
# flake8: noqa
import os

from setuptools import find_packages, setup

NAME = "torchflare"
URL = "https://github.com/Atharva-Phatak/torchflare"
EMAIL = "athp456@gmail.com"
AUTHOR = "Atharva Phatak"
REQUIRES_PYTHON = ">=3.7.0"
DESCRIPTION = (
    "TorchFlare is a simple, beginner-friendly, and easy-to-use PyTorch Framework train your models effortlessly."
)

current_file_path = os.path.abspath(os.path.dirname(__file__))


readme_file_path = os.path.join(current_file_path, "README.md")
with open(readme_file_path, "r", encoding="utf-8") as f:
    readme = f.read()


version_file_path = os.path.join(current_file_path, "version.txt")
with open(version_file_path, "r", encoding="utf-8") as f:
    version = f.read().strip()

with open(os.path.join(current_file_path, "requirements.txt"), "r") as f:
    requirements = f.read().splitlines()


setup(
    name=NAME,
    version=version,
    description=DESCRIPTION,
    long_description=readme,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    download_url=URL,
    python_requires=REQUIRES_PYTHON,
    project_urls={
        "Bug Tracker": "https://github.com/Atharva-Phatak/torchflare/issues",
        "Documentation": "https://atharva-phatak.github.io/torchflare/",
        "Source Code": "https://github.com/Atharva-Phatak/torchflare",
    },
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
    license="Apache License 2.0",
    keywords=[
        "Deep Learning",
        "Computer Vision",
        "Natural Language Processing",
        "PyTorch",
    ],
    packages=find_packages(exclude=["tests", "docs"]),
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        # Pick your license as you wish
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
