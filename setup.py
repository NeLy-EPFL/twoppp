from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()
    requirements = [l for l in requirements if not l.startswith('#')]

setup(
    name="twoppp",
    version="0.0.1",
    packages=["twoppp", "twoppp.analysis", "twoppp.behaviour", "twoppp.plot", "twoppp.register", "twoppp.utils",
              "twoppp.run",
              # "twoppp.behaviour.classification", "twoppp.behaviour.df3d", "twoppp.behaviour.fictrac",
              # "twoppp.behaviour.olfaction", "twoppp.behaviour.optic_flow", "twoppp.behaviour.synchronisation",
              # "twoppp.plot.videos", "twoppp.register.warping", "twoppp.register.warping_cluster",
              # "twoppp.utils.df", "twoppp.utils.raw_files",
              # "twoppp.denoise", "twoppp.dff", "twoppp.load",
              # "twoppp.longterm_flies", "twoppp.pipeline", "twoppp.rois"
              ],
    author="Jonas Braun",
    author_email="jonas.braun@epfl.ch",
    description="Pipeline to process simulanesouly recorded two-photon and behavioural data.",
    # long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/NeLy-EPFL/twoppp",
    install_requires=requirements,
)
