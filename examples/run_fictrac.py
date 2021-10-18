import os.path

from twoppp.load import NAS2_DIR_JB
from twoppp.behaviour.fictrac import config_and_run_fictrac

if __name__ == "__main__":
    fly_dirs = [
        os.path.join(NAS2_DIR_JB, "210930_PR_olfac", "Fly1"),
        os.path.join(NAS2_DIR_JB, "210930_PR_olfac", "Fly2"),
    ]
    for fly_dir in fly_dirs:
        config_and_run_fictrac(fly_dir)
