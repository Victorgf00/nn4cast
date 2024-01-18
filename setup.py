from setuptools import find_packages, setup

setup(
    name='nn4cast',
    packages=find_packages(include=['nn4cast']),
    version='0.1.0',
    description='Library to create NN models to forecast climate variables',
    author='Victor Galvan Fraile',
    license='MIT',
    install_requires=[''],
    classifiers=[
        'Development Status :: 1 - Production',
        'Intended Audience :: Research',
        'License :: OSI Approved :: MIT License',  # Update with your chosen license
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
