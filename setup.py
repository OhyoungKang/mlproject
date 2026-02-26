from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path)->list[str]:
    """
    this function will return the list of requirements
    """
    requirements: List[str] = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name="mlproject",
    version="0.0.1",
    description="A machine learning project template",
    author="Ohyoung Kang",
    author_email="oykang64@gmail.com",
    # license="MIT",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)