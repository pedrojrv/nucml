"""Configuration module to setup nucml's working environment."""

import os


def configure(user_path, ace_path, matlab_exe_path=""):
    """Configure an internal file necessary to enable all NucML functionalities including data loading.

    The ace_path can be an already existing directory from a serpent distribution. The
    ML_Nuclear_Data repository contaings a version of ACE files which were used to develop
    and test all functionalities. If the .ace files have different structure,
    the ace utilities may not work.

    Args:
        user_path (str): Path-like string pointing to the project directory.
        ace_path (str): Path-like string pointing to the .ace files.
        matplab_exe_path (str, optional): Path-like string pointing towards the MATLAB executable.
            The default is None

    Returns:
        None
    """
    abs_user_path = os.path.abspath(user_path)
    ame_csv_path = os.path.join(abs_user_path, "AME/CSV_Files").replace("\\", "/")
    evaluations_path = os.path.join(abs_user_path, "Evaluated_Data").replace("\\", "/")
    ensdf_path = os.path.join(abs_user_path, "ENSDF").replace("\\", "/")
    exfor_csv_path = os.path.join(abs_user_path, "EXFOR/CSV_Files").replace("\\", "/")
    bench_templ_path = os.path.join(abs_user_path, "Benchmarks/inputs/templates").replace("\\", "/")

    with open(os.path.join(os.path.dirname(__file__), 'config.py'), 'w') as f:
        f.write("ame_dir_path = r'{}' \n".format(ame_csv_path))
        f.write("evaluations_path = r'{}' \n".format(evaluations_path))
        f.write("ensdf_path = r'{}' \n".format(ensdf_path))
        f.write("exfor_path = r'{}' \n".format(exfor_csv_path))
        f.write("bench_template_path = r'{}' \n".format(bench_templ_path))
        f.write("ace_path = r'{}' \n".format(os.path.abspath(ace_path)))
        f.write("matlab_path = r'{}' \n".format(os.path.abspath(matlab_exe_path)))

    return
