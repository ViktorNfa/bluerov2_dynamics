from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
def parse_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f.readlines() if line and not line.startswith("#")]

# Read the contents of the README file for long description
def read_readme():
    with open("README.md", encoding="utf-8") as f:
        return f.read()

setup(
    name="blue_rov2py",
    version="0.1.0",
    description="A Python package for Blue ROV2 control models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Victor Nan Ayala, Gregorio Marchesini",  # Replace with your name or your organization name
    author_email="your-email@example.com",  # Replace with your email
    url="https://github.com/yourusername/blue_rov2py",  # Replace with your GitHub repo URL
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    test_suite="tests",  # If you use pytest or another test framework, adjust accordingly
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Adjust this based on your license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # You can change this depending on the Python version your package supports
)
