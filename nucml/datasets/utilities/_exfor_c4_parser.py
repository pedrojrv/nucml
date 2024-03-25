"""Heavily modified version of the parseC4.py file. Original credits given below.

Created by Caleb Mattoon on 2010-06-28.
Copyright (c) 2010 __nndc.bnl.gov__. All rights reserved.
"""

import os
import logging

from glob import glob
from pathlib import Path
from typing import Tuple

from nucml._constants import EXFOR_MATERIALS

logger = logging.getLogger(__name__)

projectiles = {
    '0': 'gammas',
    '1': 'neutrons',
    '1001': 'protons',
    '1002': 'deuterons',
    '2003': 'helions',
    '2004': 'alphas',
    'other': 'other'
}


def _get_symbol(zzz: str) -> str:
    izzz = int(zzz)
    return EXFOR_MATERIALS[-1] if izzz > 109 else EXFOR_MATERIALS[izzz]


def _read_c4(line: str) -> Tuple[str, str]:
    proj = line[:5].strip()
    targ = line[5:11].strip()
    return proj, targ


def parse_c4(xc4_file: Path, saving_dir: Path) -> None:
    """Parse EXFOR .xc4 file.

    Args:
        xc4_file (Path): Path to .xc4 file.
        saving_dir (Path): Location on which to save the parsed information.
    """
    c4 = []
    with open(xc4_file, "r") as f:
        i = 0
        for line in f:
            if line.startswith("#ENTRY "):
                i = i + 1
            c4.append(line)

    entry = []
    ENTRY = False
    data = []
    DATA = False
    ndatasets = 0
    emptySets = 0
    newline = os.linesep

    iEntry = 0
    for i in range(len(c4)):
        line = c4[i]

        # first check some special lines:
        if line.startswith("#ENTRY "):
            ENTRY = True
            entry = []
            iEntry = iEntry + 1
        if line.startswith("#DATASET "):
            if DATA:
                raise ValueError("DATASET not closed properly, line ", i)
            DATA = True
            ENTRY = False
        elif line.startswith("#TARG "):
            # grab target for each data set
            try:
                targ = line.strip().split()[1:]
                isomer = False
                if len(targ) > 1:
                    isomer = targ[1]
                zam = int(targ[0])
            except ValueError:
                print("Trouble getting target information line ", i)

        elif line.startswith("#PROJ "):
            proj = line.strip().split()[-1]
            dir = projectiles.get(proj, 'other')

        elif line.startswith("#/DATASET"):
            ndatasets += 1

            data.append(line.strip() + newline)
            data.append("#/ENTRY%s#%s#%s" % ((newline,)*3))

            # done with this data set. write to file:
            z = str(zam // 1000).zfill(3)
            a = str(zam % 1000).zfill(3)
            el = _get_symbol(z)

            if isomer:
                saving_file = saving_dir / f'{dir}/{z}_{el}_{a}_{isomer}.c4'
                f = open(saving_file, "a")
            else:
                saving_file = saving_dir / f'{dir}/{z}_{el}_{a}.c4'
                f = open(saving_file, "a")
            f.writelines(entry)
            f.writelines(data)
            f.close()
            data = []
            DATA = False

        elif line.startswith("#DATASETS   0"):
            emptySets += 1

        if not line.startswith('#'):
            PROJ, TARG = _read_c4(line)
            if not PROJ == proj and TARG == repr(zam):
                logger.info(f"Bad target/projectile information on line {i}")

        if DATA:
            data.append(line.rstrip() + newline)
        if ENTRY:
            entry.append(line.rstrip() + newline)

    logger.info(f"{ndatasets} data sets extracted of which {emptySets} are empty.")


def _sort_C4_file(filename: str) -> None:
    """Sort C4 File and save in-place.

    The sort order is MF, MT, Energy, and entry number.

    Args:
        filename (str): Path to C4 file to be sorted.
    """
    def readC4(line):
        # within the 'other' dir, still need to sort by projectile and target:
        proj = line[:5].strip()
        targ = line[5:11].strip()
        MF = int(line[12:15])
        MT = int(line[15:19])
        energy = line[22:31].strip()
        entry = line[122:127]

        try:
            energy = float(energy)
        except ValueError:
            if energy == '':
                energy = 0
            else:
                sign = energy[0]
                energy = energy[1:].replace('+', 'E+').replace('-', 'E-')
                energy = float(sign + energy)

        return proj, targ, MF, MT, energy, entry

    datasets = []
    firstPoint = True

    fin = open(filename, "rU").readlines()
    for i in range(len(fin)):
        line = fin[i]
        if line.startswith("#ENTRY"):
            firstline = i
        elif line == "\n":
            continue
        elif firstPoint and not line.startswith("#"):
            proj, targ, MF, MT, energy, entry = readC4(line)
            firstPoint = False
        elif line.startswith("#/ENTRY"):
            lastline = i + 5
            datasets.append([proj, targ, MF, MT, energy, entry, firstline, lastline + 1])
            firstPoint = True

    datasets.sort()
    fout = open(filename, "w")
    for set in datasets:
        fout.writelines(fin[set[6]:set[7]])
    fout.close()


def sort_c4_files(saving_dir: Path) -> None:
    """Sort all C4 files in the given directory and subdirectories."""
    files = []
    for projectile in projectiles.values():
        pattern = str(saving_dir / f'{projectile}/*.c4')
        files += glob(pattern)

    for f in files:
        _sort_C4_file(f)


def parse_and_sort_c4_files(xc4_file: Path, saving_dir: Path) -> None:
    for _, proj in projectiles.items():
        proj_dir = saving_dir / proj
        proj_dir.mkdir(exist_ok=True)

    parse_c4(xc4_file, saving_dir)
    sort_c4_files(saving_dir)
