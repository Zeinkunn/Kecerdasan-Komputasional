from setuptools import setup, find_packages

setup(
    name="ml-project",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'pandas',
        'numpy',
        'scikit-learn',
        'kagglehub',
        'matplotlib'
    ],
    python_requires='>=3.13',
    author="Zainul Mutawakkil",
    description="A machine learning project with required dependencies",
)

