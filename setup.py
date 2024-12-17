from setuptools import setup, find_packages

setup(
    name="federated-rag-financial",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.6.0",
        "tensorflow-federated>=0.20.0",
        "torch>=1.9.0",
        "transformers>=4.11.0",
        "elasticsearch>=7.0.0",
        "faiss-cpu>=1.7.0",
        "pandas>=1.3.0",
        "numpy>=1.19.0",
        "pytest>=6.0.0",
        "black>=21.0.0",
        "flake8>=3.9.0",
        "sphinx>=4.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Federated Learning for Privacy-Preserving Financial Data Generation with RAG Integration",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/federated-rag-financial",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8",
)