from setuptools import setup, find_packages

setup(
    name="dockstream-studio",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "biopython==1.81",
        "rdkit==2023.9.5",
        "meeko==0.5.0",
        "stmol==0.0.9",
        "py3Dmol==2.0.4",
        "xhtml2pdf==0.2.11",
        "selenium==4.18.1",
        "webdriver-manager==4.0.1",
    ],
)
