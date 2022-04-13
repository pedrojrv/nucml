"""Setup script for building, distribution and installation of nucml."""

import setuptools
import pathlib

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

HERE = pathlib.Path(__file__).parent

setuptools.setup(
    name="nucml",
    version="1.0.5.dev1",
    author="Pedro Junior Vicente Valdez",
    author_email="pedro.vicentevz@berkeley.edu",
    description="ML-oriented tools for navigating the nuclear data evaluation pipeline.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pedrojrv/nucml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",

    ],
    keywords="nuclear ml nucml nuclearml nucleardata data cross section",
    project_urls={
        'Documentation': 'https://pedrojrv.github.io/nucml'
    },
    license="GNU General Public License v3 or later (GPLv3+)",
    include_package_data=True,
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'nucml-configure=nucml.configure:main',
        ]
    }
)
