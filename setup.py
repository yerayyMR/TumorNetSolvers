from setuptools import setup, find_packages

# Utility function to read requirements.txt
def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]

setup(
    name="TumorNetSolvers", 
    version="1.0.0", 
    author_email="zeineb.haouari@tum.de",
    description="NN-based forward solvers for tumor growth modeling",
    packages=find_packages(where="src"), 
    package_dir={"": "src"},  
    install_requires=parse_requirements("requirements.txt"), 
    python_requires=">=3.7", 
    license="Apache License 2.0"
)
