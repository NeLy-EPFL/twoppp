from setuptools import setup, find_packages

try:
    import utils2p
except ImportError or ModuleNotFoundError:
    raise ImportError("utils2p must be installed in environment.")

try:
    from deepfly.CameraNetwork import CameraNetwork
except ImportError or ModuleNotFoundError:
    raise ImportError("deepfly must be installed in environment.")

try:
    from deepinterpolation import interface as denoise

except ImportError or ModuleNotFoundError:
    raise ImportError("DeepInterpolation must be installed in environment.")

try:
    import ofco
except ImportError or ModuleNotFoundError:
    raise ImportError("ofco must be installed in environment.")

try:
    import utils_video
except ImportError or ModuleNotFoundError:
    raise ImportError("utils_video must be installed in environment.")

try:
    import df3dPostProcessing
except ImportError or ModuleNotFoundError:
    raise ImportError("df3dPostProcessing must be installed in environment.")

try:
    import behavelet
except ImportError or ModuleNotFoundError:
    raise ImportError("behavelet must be installed in environment.")

with open("README.md", "r") as fh:
    long_description = fh.read()

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
)
