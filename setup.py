import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nucml", # Replace with your own username
    version="1.0.0.dev1",
    author="Pedro Jr Vicente Valdez",
    author_email="vicentepedrojr@gmail.com",
    description="ML-oriented tools for navingating the nuclear data evaluation pipeline.",
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
    install_requires=["natsort", "xgboost", "pandas", "plotly", "matplotlib", "scikit-learn", "seaborn", "imageio"], 
    python_requires='>=3.6',
)


