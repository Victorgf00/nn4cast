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
        'numpy==1.23.5', 'matplotlib==3.7.1', 'xarray==2023.2.0', 'cartopy==0.21.1', 'pandas==1.5.3', 'xskillscore==0.0.24', 'scipy==1.10.1', 'tensorflow==2.10.0', 'keras==2.10.0', 'kerastuner==1.3.4', 'scikit-learn==1.2.1'
 ],
    python_requires=">=3.6",
)
