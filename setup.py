from setuptools import setup, find_packages

setup(
    name='comfy-cli',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'comfy = .module:main_func'
        ]
    }
)