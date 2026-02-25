from pathlib import Path
from scipy.io import loadmat

def find_data_files(folder):
    """
    Find all supported data files in a folder.

    Parameters
    ----------
    folder : str or Path
        Folder containing measurement files.

    Returns
    -------
    list of Path
        Sorted list of file paths.
    """

    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    supported_extensions = [".mat", ".c3d", ".csv", ".txt"]

    files = []

    for file in folder.rglob("*"):

        if file.is_file() and file.suffix.lower() in supported_extensions:
            files.append(file)

    return sorted(files)




def load_mat(path):
    """
    Load a MATLAB .mat file and return the content as dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MAT file not found: {path}")

    mat = loadmat(path, squeeze_me=True, struct_as_record=False)

    mat.pop("__header__", None)
    mat.pop("__version__", None)
    mat.pop("__globals__", None)

    return mat


def get_parameters_r(mat):
    """
    Extract PARAMETERS.R from a loaded mat dict.
    """
    if "PARAMETERS" not in mat:
        raise KeyError("Top-level key 'PARAMETERS' not found in .mat file.")

    params = mat["PARAMETERS"]

    if isinstance(params, dict):
        if "R" not in params:
            raise KeyError("Key 'R' not found inside PARAMETERS.")
        return params["R"]

    if hasattr(params, "R"):
        return getattr(params, "R")

    raise TypeError(f"PARAMETERS has unexpected type: {type(params)}")


def list_variable_names_from_r(r_obj):
    """
    List available variable names inside PARAMETERS.R.
    """
    if isinstance(r_obj, dict):
        return sorted(r_obj.keys())

    if hasattr(r_obj, "__dict__"):
        return sorted(r_obj.__dict__.keys())

    if hasattr(r_obj, "dtype") and getattr(r_obj.dtype, "names", None):
        return sorted(list(r_obj.dtype.names))

    raise TypeError(f"R has unexpected type: {type(r_obj)}")


def list_mat_variable_names(mat_path):
    """
    Load a .mat file and return variable names from PARAMETERS.R.
    """
    mat = load_mat(mat_path)
    r_obj = get_parameters_r(mat)
    return list_variable_names_from_r(r_obj)


def save_variable_names_to_txt(variable_names, output_path):
    """
    Save variable names into a text file.

    Parameters
    ----------
    variable_names : list[str]
        List of variable names

    output_path : str or Path
        Output text file
    """

    output_path = Path(output_path)

    with open(output_path, "w", encoding="utf-8") as f:

        f.write("Variablen in PARAMETERS.R:\n")
        f.write("---------------------------\n\n")

        for name in variable_names:
            f.write(name + "\n")


def export_mat_structure_to_txt(mat_path, output_txt):
    """
    Export structure of a .mat file into a text file.

    Parameters
    ----------
    mat_path : str or Path
        Path to MAT file

    output_txt : str or Path
        Output text file
    """

    mat = load_mat(mat_path)

    output_txt = Path(output_txt)

    with open(output_txt, "w", encoding="utf-8") as f:

        f.write("MAT FILE STRUCTURE\n")
        f.write("===================\n\n")

        for key in mat.keys():

            f.write(f"TOP LEVEL: {key}\n")
            f.write("-----------------\n")

            obj = mat[key]

            f.write(f"Type: {type(obj)}\n")

            # Shape wenn vorhanden
            if hasattr(obj, "shape"):
                f.write(f"Shape: {obj.shape}\n")

            f.write("\n")

            # MATLAB struct → Felder anzeigen
            if hasattr(obj, "__dict__"):

                f.write("Fields:\n")

                for subkey in obj.__dict__.keys():

                    subobj = getattr(obj, subkey)

                    f.write(f"  {subkey}\n")
                    f.write(f"    Type: {type(subobj)}\n")

                    if hasattr(subobj, "shape"):
                        f.write(f"    Shape: {subobj.shape}\n")

                    f.write("\n")

            f.write("\n\n")


def export_top_level_fields_to_txt(mat_path, output_dir):
    """
    For each top-level variable in a MAT file, export its field names into
    separate .txt files.

    Example outputs:
    - CONTACT_vars.txt
    - FP_vars.txt
    - PARAMETERS_vars.txt

    Parameters
    ----------
    mat_path : str or Path
        Path to the .mat file
    output_dir : str or Path
        Folder where txt files will be written
    """
    mat_path = Path(mat_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mat = load_mat(mat_path)

    for top_key, obj in mat.items():
        fields = _get_field_names(obj)

        out_file = output_dir / f"{top_key}_vars.txt"

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"MAT file: {mat_path.name}\n")
            f.write(f"TOP LEVEL: {top_key}\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"Python type: {type(obj)}\n")
            if hasattr(obj, "shape"):
                f.write(f"Shape: {getattr(obj, 'shape')}\n")

            f.write("\nFIELDS / VARIABLE NAMES:\n")
            f.write("-" * 40 + "\n")

            if fields:
                for name in fields:
                    f.write(name + "\n")
            else:
                f.write("(no fields found — this object may be a plain array/value)\n")




from pathlib import Path

import numpy as np
from scipy.io import loadmat


def _get_field_names(obj) -> list[str]:
    """
    Return field/variable names of a MATLAB-like object.

    This handles:
    - MATLAB structs loaded by scipy (mat_struct) -> via __dict__ / _fieldnames
    - dict -> keys()
    - numpy structured arrays -> dtype.names
    If no fields exist, returns an empty list.
    """
    # Case 1: dict-like
    if isinstance(obj, dict):
        return sorted([str(k) for k in obj.keys()])

    # Case 2: scipy MATLAB struct objects (often have __dict__)
    if hasattr(obj, "_fieldnames") and isinstance(getattr(obj, "_fieldnames"), list):
        # _fieldnames contains actual MATLAB fields (good)
        return sorted([str(n) for n in getattr(obj, "_fieldnames") if str(n) != "_fieldnames"])

    if hasattr(obj, "__dict__"):
        # fallback: read attributes, ignore internal ones
        names = [k for k in obj.__dict__.keys() if not k.startswith("_")]
        return sorted([str(n) for n in names if str(n) != "_fieldnames"])

    # Case 3: numpy structured array
    if isinstance(obj, np.ndarray) and obj.dtype is not None and obj.dtype.names:
        return sorted([str(n) for n in obj.dtype.names])

    return []

def export_deep_variable_names(mat_path, output_dir):
    """
    Export variable names one level deeper for each top-level structure.

    Example:

    PARAMETERS_R_variables.txt
    CONTACT_variables.txt
    FP_variables.txt
    """

    mat = load_mat(mat_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for top_key, obj in mat.items():

        # MATLAB struct erwartet
        if not hasattr(obj, "__dict__"):
            continue

        subkeys = list(obj.__dict__.keys())

        if not subkeys:
            continue

        first_subkey = subkeys[0]

        sub_obj = getattr(obj, first_subkey)

        # nächste Ebene Variablen holen
        if hasattr(sub_obj, "__dict__"):
            var_names = sorted(sub_obj.__dict__.keys())
        else:
            continue

        out_file = output_dir / f"{top_key}_variables.txt"

        with open(out_file, "w", encoding="utf-8") as f:

            f.write(f"{top_key} VARIABLES\n")
            f.write("--------------------\n\n")

            f.write(f"Based on: {first_subkey}\n\n")

            for v in var_names:
                f.write(v + "\n")


def export_real_variable_names(mat_path, output_dir):
    """
    Export variable names for each top-level structure.

    Structure:

    TOPLEVEL
        TrialName
            Variables

    Example output files:

    CONTACT_variables.txt
    FP_variables.txt
    PARAMETERS_variables.txt
    """

    mat = load_mat(mat_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for top_key, obj in mat.items():

        # Nur MATLAB structs betrachten
        if not hasattr(obj, "__dict__"):
            continue

        # erste Ebene Felder holen
        level1_names = [n for n in obj.__dict__.keys() if n != "_fieldnames"]

        if len(level1_names) == 0:
            continue

        # erstes Feld nehmen (Trial oder R)
        level1_name = level1_names[0]

        level1_obj = getattr(obj, level1_name)

        # Variablen holen
        if not hasattr(level1_obj, "__dict__"):
            continue

        var_names = sorted(
            [n for n in level1_obj.__dict__.keys() if n != "_fieldnames"]
        )

        # TXT Datei schreiben
        out_file = output_dir / f"{top_key}_variables.txt"

        with open(out_file, "w", encoding="utf-8") as f:

            f.write(f"TOP LEVEL: {top_key}\n")
            f.write(f"Based on: {level1_name}\n")
            f.write("\nVARIABLES:\n\n")

            for v in var_names:
                f.write(v + "\n")



def extract_dual_axis_parameters(mat):
    """
    Extract CONTACT parameters needed for DF and SF_norm.

    Returns
    -------
    dict
    """

    contact = mat["CONTACT"]

    trial_names = [k for k in contact.__dict__.keys() if not k.startswith("_")]
    trial = getattr(contact, trial_names[0])

    contact_times = getattr(trial, "ContactTimes")
    flight_times = getattr(trial, "FlightTimes")
    step_frequency = getattr(trial, "StepFrequency")

    return dict(
        ContactTimes=contact_times,
        FlightTimes=flight_times,
        StepFrequency=step_frequency
    )

from pathlib import Path


def parse_filename_info(mat_path):
    """
    Extract subject and kilometer from filename.

    Supports:

    P61_km01_0.mat → 1.0
    P61_km01_5.mat → 1.5
    P61_km10_0.mat → 10.0
    """

    name = Path(mat_path).stem

    parts = name.split("_")

    subject = parts[0]

    km_integer = parts[1].replace("km", "")
    km_decimal = parts[2]

    km = float(f"{int(km_integer)}.{km_decimal}")

    return dict(
        subject=subject,
        km=km
    )