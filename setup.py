from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="multivae",
    version="1.0.2",
    author="Agathe Senellart, Clement Chadebec",
    author_email="agathe.senellart@inria.fr",
    description="Unifying Generative Multimodel Variational Autoencoders in Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AgatheSenellart/MultiVae",
    project_urls={"Bug Tracker": "https://github.com/AgatheSenellart/MultiVae/issues"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pythae",
        "cloudpickle>=2.1.0",
        "numpy>=1.19",
        "pydantic>=2.0",
        "scikit-learn",
        "scipy>=1.7.1",
        "torch>=1.10.1",
        "tqdm",
        "typing_extensions",
        "dataclasses>=0.6",
        "torchvision",
        "pandas",
        "nltk",
        "torchmetrics",
        "matplotlib",
    ],
    python_requires=">=3.8",
)
