from setuptools import setup

setup(
    name="nobrainer-zoo",
    version='0.0.1',
    py_modules=['cli'],
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        nobrainer-zoo=cli:gpu_run
    ''',
)