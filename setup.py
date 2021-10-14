from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="twoppp",
    version="0.0.1",
    packages=["twoppp", "utils2p.external", "utils2p.external.tifffile"],
    author="Jonas Braun",
    author_email="jonas.braun@epfl.ch",
    description="Pipeline to process simulanesouly recorded two-photon and behavioural data.",
    # long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/NeLy-EPFL/twoppp",
)
