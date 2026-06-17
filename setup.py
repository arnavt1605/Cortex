from setuptools import setup, find_packages

setup(
    name="cortex-memory",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ollama",
        "sentence-transformers",
        "numpy",
        "pycryptodome"
    ],
    entry_points={
        'console_scripts': [
            'cortexdb=cortex.chat:main', 
        ],
    },
)