from setuptools import setup
from pip.req import parse_requirements
from fgem import __version__

install_reqs = parse_requirements("requirements.txt")
reqs = [str(ir.req) for ir in install_reqs]

setup(
    name='fgem',
    version=__version__,

    url='https://github.com/aljubrmj/FGEM',
    author='Mohammad Aljubran',
    author_email='aljubrmj@gmail.com; aljubrmj@stanford.edu',
    install_requires=reqs,
    py_modules=['fgem'],
)