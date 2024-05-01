from setuptools import setup, find_packages
from fgem import __version__

setup(
    name='fgem',
    version=__version__,

    url='https://github.com/aljubrmj/FGEM',
    author='Mohammad Aljubran',
    author_email='aljubrmj@gmail.com; aljubrmj@stanford.edu',

    install_requires=[
                    "matplotlib>=3.6.3",
                    "numpy>=1.25.2",
                    "pandas>=1.5.2",
                    "pyXSteam>=0.4.9",
                    "scipy>=1.11.3",
                    "seaborn>=0.12.2",
                    "Shapely>=2.0.1",
                    "tqdm>=4.64.1",
                    "numpy-financial>=1.0.0",
                    "timezonefinder>=6.2.0",
                    "meteostat>=1.6.5",
                    "geopandas>=0.13.0",
                    "s3fs>=2024.3.1",
                    "zarr>=2.17.2",
                    "fsspec>=2024.3.1",
                    "kerchunk>=0.2.4"],

    packages=find_packages(),
    py_modules=['fgem'],
    include_package_data=True,
    package_data={'fgem': ['data/*']}
)