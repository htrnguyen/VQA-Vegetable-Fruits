from setuptools import setup, find_packages

setup(
    name="vqa-fruits",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
        "pyyaml",
        "tqdm",
        "scikit-learn",
    ],
)
