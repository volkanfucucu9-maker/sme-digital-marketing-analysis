from setuptools import setup, find_packages

setup(
    name="sme-digital-marketing-analysis",
    version="1.0.0",
    author="Volkan Fuucu",
    author_email="volkanfucucu9@gmail.com",
    description="SME Digital Marketing Effectiveness Analysis - MSc Dissertation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/volkanfucucu9-maker/sme-digital-marketing-analysis",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0", 
        "statsmodels>=0.12.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0"
    ],
)
