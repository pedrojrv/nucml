"""Data modules to aid manipulation of Serpent related files."""
import os
import scipy.io
import pandas as pd
import shutil
from pathlib import Path

import nucml.config as config
import nucml.general_utilities as gen_utils

matlab_path = config.matlab_path
template_path = config.bench_template_path


def copy_benchmark_files(benchmark_name, saving_dir):
    """Copy all files for a given benchmark from the benchmark repository to a given directory.

    Args:
        benchmark_name (str): Benchmark name. Check repository for valid names.
        saving_dir (str): Path-like string where the new benchmark files will be saved to.

    Returns:
        None
    """
    to_replace = "to_replace"
    new_file_path = os.path.join(saving_dir, "sss_endfb7u.xsdata")
    to_insert = os.path.abspath(new_file_path).replace("C:\\", "/mnt/c/").replace("\\", "/")

    benchmark_path = os.path.join(template_path, benchmark_name + "/input")
    new_benchmark_path = os.path.join(saving_dir, "input")

    with open(benchmark_path, "rt") as benchmark_file, open(new_benchmark_path, "wt") as new_benchmark_file:
        for line in benchmark_file:
            new_benchmark_file.write(line.replace(to_replace, to_insert))

    gen_utils.convert_dos_to_unix(new_benchmark_path)
    shutil.copyfile(os.path.join(template_path, "converter.m"), os.path.join(saving_dir, "converter.m"))


def gather_benchmark_results(searching_directory):
    """Gathers all benchmark results from the resulting .mat files in the searching_directory and all subdirectories.

    Args:
        searching_directory (str): Path to directory to search for .mat files.

    Returns:
        DataFrame: Contains results for all found .mat files.
    """
    all_results, names, benchmark_names, k_results_ana, k_unc_ana, k_results_imp, k_unc_imp = ([] for i in range(7))
    for root, _, files in os.walk(searching_directory):
        mat_files = [file for file in files if file.endswith(".mat")]
        for file in mat_files:
            name_to_append = os.path.basename(Path(root).parents[0])
            names.append(name_to_append)
            all_results.append(os.path.abspath(os.path.join(root, file)))
            benchmark_names.append(os.path.basename(os.path.dirname(os.path.abspath(os.path.join(root, file)))))

    for mat_file in all_results:
        mat = scipy.io.loadmat(mat_file)
        k_results_ana.append(mat["ANA_KEFF"][0][0])
        k_unc_ana.append(mat["ANA_KEFF"][0][1])
        k_results_imp.append(mat["IMP_KEFF"][0][0])
        k_unc_imp.append(mat["IMP_KEFF"][0][1])

    results_df = pd.DataFrame({
        "Model": names, "Benchmark": benchmark_names, "K_eff_ana": k_results_ana, "Unc_ana": k_unc_ana,
        "K_eff_imp": k_results_imp, "Unc_imp": k_unc_imp})
    for k_type in ['Ana', 'Imp']:
        results_df[f"Deviation_{k_type}"] = results_df[[f'K_eff_{k_type.lower()}']].apply(lambda k: abs((k-1)/1))
    return results_df


def generate_serpent_bash(searching_directory, script_name, benchmark="all", omp=10):
    """Generate bash script to run all experiments.

    Gather the path to all "input" benchmark files and returns a single bash script to run all
    Serpent simulations and convert the resulting matlab file into .mat files for later reading.

    Args:
        searching_directory (str): Top level directory that will be searched for "input" files.

    Returns:
        None
    """
    all_serpent_files = []
    all_serpent_files_linux = []

    for root, _, files in os.walk(searching_directory):
        if benchmark not in ["all", root]:
            raise ValueError("Benchmark value not supported.")

        input_files = [os.path.abspath(os.path.join(root, file)) for file in files if file.endswith("input")]
        input_files = [file for file in input_files if "template" not in file]
        all_serpent_files.extend(input_files)

    for i in all_serpent_files:
        new = i.replace("C:\\", "/mnt/c/").replace("\\", "/")
        all_serpent_files_linux.append("cd {}".format(os.path.dirname(new)) + "/")
        all_serpent_files_linux.append("sss2 -omp {} ".format(omp) + os.path.basename(new))
        all_serpent_files_linux.append(
            matlab_path + " -nodisplay -nosplash -nodesktop -r \"run('converter.m');exit;\" ".replace("\\", ""))

    script_path = os.path.join(searching_directory, '{}.sh'.format(script_name))

    with open(script_path, 'w') as f:
        for item in all_serpent_files_linux:
            f.write("%s\n" % item)

    gen_utils.convert_dos_to_unix(script_path)
