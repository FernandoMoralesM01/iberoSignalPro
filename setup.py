from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name='iberoSignalPro',
    version='0.2',
    description= "A package to process signals and images",
    packages=find_packages(),
    author="Fernando Morales Magallon, Erik Rene Bojorges, Pablo Roca Mendoza",
    url = "https://github.com/FernandoMoralesM01/EEG-Voluntary-Involuntary-Movements",
    install_requires=requirements,
    python_requires=">=3.10",
    # Agrega otros metadatos como author, description, etc. 
)