from setuptools import setup, find_packages

setup(
        name='project2',
        version='1.0',
        author='Lance Ensminger',
        authour_email='lance.ensminger@ou.edu',
        packages=find_packages(exclude=('tests', 'docs')),
        setup_requries=['pytest-runner'],
        tests_require=['pytest']
)
