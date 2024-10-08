import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('setup.cfg', 'r') as f:
    content = f.read()
    start = content.find('version = ')
    end = content[start:].find('\n') + start
    version = content[start:end].split(' = ')[-1]
    print(f'[INFO] Running on version {version}')

setuptools.setup(
    name="nn4cast",
    author="Victor Galvan Fraile",
    author_email="vgalvanfraile@gmail.com",
    description="Python API for applying methodologies to .nc Datasets and AI methodologies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Victorgf00/nn4cast",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where=""),
    install_requires=[
        'numpy==1.26.4', 'matplotlib==3.8.4', 'xarray==2024.3.0', 'pandas==1.5.3', 'xskillscore==0.0.26', 'scipy==1.10.1', 'tensorflow==2.7.0', 'keras==2.7.0', 'keras-tuner==1.0.2', 'scikit-learn==1.0.2', 'alibi==0.7.0', 'netcdf4==1.6.2', 'protobuf==3.20.3'
 ],
    python_requires="<=3.9.19",
)  
