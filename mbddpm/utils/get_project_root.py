from pathlib import Path


def get_project_root():

    current_file = Path(__file__).resolve()

    project_root = current_file.parents[2]

    return project_root
