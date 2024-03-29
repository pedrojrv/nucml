"""Configuration module to setup nucml's working environment."""

import os
import argparse
from pathlib import Path


def configure(user_path, ace_path, matlab_exe_path=""):
    """Configure an internal file necessary to enable all NucML functionalities including data loading.

    The ace_path can be an already existing directory from a serpent distribution. The ML_Nuclear_Data repository
    contains a version of ACE files which were used to develop and test all functionalities. If the .ace files have
    a different structure, the ace utilities are not guaranteed work.

    Args:
        user_path (str): Path-like string pointing to the project directory.
        ace_path (str): Path-like string pointing to the .ace files.
        matplab_exe_path (str, optional): Path-like string pointing towards the MATLAB executable.
            The default is None

    Returns:
        None
    """
    abs_user_path = Path(user_path).resolve()
    ame_csv_path = abs_user_path / "AME/CSV_Files"
    evaluations_path = abs_user_path / "Evaluated_Data"
    ensdf_path = abs_user_path / "ENSDF"
    exfor_csv_path = abs_user_path / "EXFOR/CSV_Files"
    bench_templ_path = abs_user_path / "Benchmarks/inputs/templates"

    for path in [abs_user_path, ame_csv_path, evaluations_path, ensdf_path, exfor_csv_path, bench_templ_path]:
        if not path.is_dir():
            raise FileExistsError(f"{path} is not a directory. Make sure it exists.")

    with open(os.path.join(os.path.dirname(__file__), 'config.py'), 'w') as f:
        f.write('"""File automatically generated by nucml-configure. This file can be amended if needed."""\n')
        f.write("ame_dir_path = r'{}'\n".format(ame_csv_path))
        f.write("evaluations_path = r'{}'\n".format(evaluations_path))
        f.write("ensdf_path = r'{}'\n".format(ensdf_path))
        f.write("exfor_path = r'{}'\n".format(exfor_csv_path))
        f.write("bench_template_path = r'{}'\n".format(bench_templ_path))
        f.write("ace_path = r'{}'\n".format(os.path.abspath(ace_path)))
        f.write("matlab_path = r'{}'\n".format(os.path.abspath(matlab_exe_path)))


def main():  # noqa
    parser = argparse.ArgumentParser(description='Configures paths for nucml to access nuclear data.')
    parser.add_argument("-p", "--path-to-data", action='store', dest='path_to_data', help='Path to nuclear data.')
    parser.add_argument("-a", "--path-to-ace", action='store', dest='path_to_ace', help='Path to ace data.')
    parser.add_argument(
        "-m", "--path_to_matlab", action='store', dest='path_to_matlab', default="", help='Path to matlab exe.')
    args = parser.parse_args()

    configure(args.path_to_data, args.path_to_ace, args.path_to_matlab)


if __name__ == "__main__":
    main()
