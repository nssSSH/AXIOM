from setuptools import setup, find_packages

setup(
    name="axiom-lang",
    version="0.1.0",
    description="AXIOM v0.1 — Adaptive eXpressive Intent-Oriented Matrix Language compiler",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="AXIOM Project",
    python_requires=">=3.9",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "axiom=axiom.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Compilers",
    ],
)
