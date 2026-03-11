"""Setup script for quantum-sensing-routing package."""

from setuptools import setup, find_packages

setup(
    name='quantum-sensing-routing',
    version='1.0.0',
    description=(
        'Deep Reinforcement Learning-Driven Autonomous Entanglement Routing '
        'in Distributed Variational Quantum Sensing Networks'
    ),
    author='Research Team',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'gymnasium>=0.28.0',
        'matplotlib>=3.7.0',
        'pyyaml>=6.0',
        'networkx>=3.1',
        'tqdm>=4.65.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.3.0',
            'tensorboard>=2.13.0',
            'seaborn>=0.12.0',
            'pandas>=2.0.0',
            'scikit-learn>=1.2.0',
        ],
    },
)
