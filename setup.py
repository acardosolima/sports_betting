from setuptools import setup, find_packages

setup(
    name="Sports Betting",
    use_scm_version=True,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    python_requires=">=3.10",
)
