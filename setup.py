from setuptools import setup, find_packages

setup(
    name="quantum_ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "qiskit>=2.0.0",
        "qiskit-aer>=0.17.0",
        "matplotlib>=3.5.0",
        "numpy>=1.22.0",
        "tensorflow>=2.10.0",
        "gym>=0.21.0",
        "pandas>=1.4.0",
        "networkx>=2.7.0",
        "scipy>=1.8.0",
        "pyqubo>=1.4.0",
        "torch>=2.0.0"
    ],
    author="Zayaan",
    author_email="smohamedzayaan@gmail.com",
    description="AI-Driven Quantum Computing Framework for Solving NP-Hard Problems",
)

