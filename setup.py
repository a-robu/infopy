from setuptools import setup

setup(
    name='infopy',
    version='1.8.0',
    py_modules=['infopy', 'labeled_matrix'],
    install_requires=[
        'pytest==3.2.2',
        'pylint==1.7.4',
        'pymatrix==3.0.0'
    ]
)
