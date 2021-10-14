import os.path

FILE_PATH = os.path.realpath(__file__)
TWOPPP_PATH, _ = os.path.split(FILE_PATH)
MODULE_PATH, _ = os.path.split(TWOPPP_PATH)
OUTPUT_PATH = os.path.join(MODULE_PATH, "outputs")
TMP_PATH = os.path.join(MODULE_PATH, "tmp")
