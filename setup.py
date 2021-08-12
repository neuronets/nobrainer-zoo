from setuptools import setup, find_packages

setup(
    name="nobrainer-zoo",
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
        'pyyaml'
    ],
    entry_points='''
        [console_scripts]
        nobrainer-zoo=nobrainerzoo.cli:cli
    ''',
)