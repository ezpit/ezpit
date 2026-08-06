from collections import Counter
import numpy as np
import os


# helpers gathered from EZPIT

def parse_composition(composition):
    """
    Parse a composition string from the Control Panel and turns it into a dictionary.

    Example:
        'Si 1 O 2' --> {'Si': 1, 'O': 2}

    Assumes that elements and quantities alternate, separated by spaces.
    """

    if composition == "":
        raise ValueError("The composition field must not be empty")
    tokens = composition.strip().split()
    if len(tokens) % 2 != 0:
        raise ValueError("Invalid composition string. Must contain pairs of elements and quantities.")

    composition_dict = Counter()
    for i in range(0, len(tokens), 2):
        element = tokens[i]
        try:
            quantity = int(tokens[i + 1])
        except ValueError:
            raise ValueError(f"Invalid quantity for element '{element}': {tokens[i + 1]}")
        composition_dict[element] = quantity

    return composition_dict


# Element symbol -> full name. Used to disambiguate easily-confused symbols
# (e.g. 'C' carbon vs 'Co' cobalt) in the Composition preview label.
ELEMENT_FULL_NAMES = {
    'H': 'Hydrogen', 'He': 'Helium', 'Li': 'Lithium', 'Be': 'Beryllium',
    'B': 'Boron', 'C': 'Carbon', 'N': 'Nitrogen', 'O': 'Oxygen',
    'F': 'Fluorine', 'Ne': 'Neon', 'Na': 'Sodium', 'Mg': 'Magnesium',
    'Al': 'Aluminium', 'Si': 'Silicon', 'P': 'Phosphorus', 'S': 'Sulfur',
    'Cl': 'Chlorine', 'Ar': 'Argon', 'K': 'Potassium', 'Ca': 'Calcium',
    'Sc': 'Scandium', 'Ti': 'Titanium', 'V': 'Vanadium', 'Cr': 'Chromium',
    'Mn': 'Manganese', 'Fe': 'Iron', 'Co': 'Cobalt', 'Ni': 'Nickel',
    'Cu': 'Copper', 'Zn': 'Zinc', 'Ga': 'Gallium', 'Ge': 'Germanium',
    'As': 'Arsenic', 'Se': 'Selenium', 'Br': 'Bromine', 'Kr': 'Krypton',
    'Rb': 'Rubidium', 'Sr': 'Strontium', 'Y': 'Yttrium', 'Zr': 'Zirconium',
    'Nb': 'Niobium', 'Mo': 'Molybdenum', 'Tc': 'Technetium', 'Ru': 'Ruthenium',
    'Rh': 'Rhodium', 'Pd': 'Palladium', 'Ag': 'Silver', 'Cd': 'Cadmium',
    'In': 'Indium', 'Sn': 'Tin', 'Sb': 'Antimony', 'Te': 'Tellurium',
    'I': 'Iodine', 'Xe': 'Xenon', 'Cs': 'Caesium', 'Ba': 'Barium',
    'La': 'Lanthanum', 'Ce': 'Cerium', 'Pr': 'Praseodymium', 'Nd': 'Neodymium',
    'Pm': 'Promethium', 'Sm': 'Samarium', 'Eu': 'Europium', 'Gd': 'Gadolinium',
    'Tb': 'Terbium', 'Dy': 'Dysprosium', 'Ho': 'Holmium', 'Er': 'Erbium',
    'Tm': 'Thulium', 'Yb': 'Ytterbium', 'Lu': 'Lutetium', 'Hf': 'Hafnium',
    'Ta': 'Tantalum', 'W': 'Tungsten', 'Re': 'Rhenium', 'Os': 'Osmium',
    'Ir': 'Iridium', 'Pt': 'Platinum', 'Au': 'Gold', 'Hg': 'Mercury',
    'Tl': 'Thallium', 'Pb': 'Lead', 'Bi': 'Bismuth', 'Po': 'Polonium',
    'At': 'Astatine', 'Rn': 'Radon', 'Fr': 'Francium', 'Ra': 'Radium',
    'Ac': 'Actinium', 'Th': 'Thorium', 'Pa': 'Protactinium', 'U': 'Uranium',
    'Np': 'Neptunium', 'Pu': 'Plutonium',
}

# Cache of valid element symbols loaded from the form-factor database.
_VALID_ELEMENTS_CACHE = None


def _load_valid_elements():
    """Load the set of element symbols that exist in the form-factor database
    (ezpit.elem_tables.AFF_ELEMENTS). Cached after first load."""
    global _VALID_ELEMENTS_CACHE
    if _VALID_ELEMENTS_CACHE is not None:
        return _VALID_ELEMENTS_CACHE
    from ezpit.elem_tables import AFF_ELEMENTS
    _VALID_ELEMENTS_CACHE = set(AFF_ELEMENTS)
    return _VALID_ELEMENTS_CACHE


def preview_composition(text):
    """
    Build a human-readable preview of a composition string for the Control Panel.

    Returns a dict:
        {
          'ok':      bool,    # False if there is a blocking error
          'message': str,     # preview text (ok) or error text (not ok)
          'parsed':  dict,    # {symbol: count} parsed so far
        }

    Behaviour:
      - Valid elements are NEVER rejected or auto-changed. 'C' (carbon) and
        'Co' (cobalt) both pass; the preview shows the full element name so the
        user can catch a symbol mix-up themselves.
      - Only genuinely unknown symbols (not in the form-factor database) are
        flagged as errors.
      - If the database file is unavailable, existence checking is skipped and
        the preview is still produced.
    """
    if text is None or text.strip() == "":
        return {'ok': True, 'message': "", 'parsed': {}}

    tokens = text.strip().split()
    if len(tokens) % 2 != 0:
        return {'ok': False,
                'message': "Enter element/count pairs, e.g. Co 38 O 119",
                'parsed': {}}

    valid = _load_valid_elements()  # may be None -> skip existence check
    parsed = {}
    errors = []
    for i in range(0, len(tokens), 2):
        el, qty = tokens[i], tokens[i + 1]
        try:
            n = int(qty)
        except ValueError:
            errors.append(f"'{el}': count '{qty}' is not an integer")
            continue
        if n <= 0:
            errors.append(f"'{el}': count must be positive")
            continue
        if valid is not None and el not in valid:
            errors.append(f"'{el}' is not a known element")
            continue
        parsed[el] = parsed.get(el, 0) + n

    if errors:
        return {'ok': False, 'message': "  •  ".join(errors), 'parsed': parsed}

    parts = [f"{el} ({ELEMENT_FULL_NAMES.get(el, '?')}) {n}"
             for el, n in parsed.items()]
    total = sum(parsed.values())
    message = " ,   ".join(parts) + f"      ({total} atoms)"
    return {'ok': True, 'message': message, 'parsed': parsed}


def convert_atom_names(composition: dict) -> list[str]:
    """
    if composition = {'Co': 3, 'O': 4, 'P': 1},
    atom_names = losa.convert_atom_names(composition)
    print('atom_names = ', atom_names)
    --> atom_names = ['Co', 'Co', 'Co', 'O', 'O', 'O', 'O', 'P']
    Returns
    """
    return [el for el, count in composition.items() for _ in range(count)]


def group_atoms(atom_names):
    """
    Get unique atom names, their counts, and the index of atom in the unique
    name list. Results will be used by other functions.
            # if composition = {'Co':2, 'O':1, 'P':1},
            counter =  Counter({'Co': 2, 'O': 1, 'P': 1})
            atom_unique_names =  ['Co', 'O', 'P']
            atom_counts =  [2 1 1]  # "2" is for 'Co', "1" is for 'O', "1" is for 'P'
            atom_indices  =  [0 0 1 2] #
    """
    counter = Counter(atom_names)
    atom_uni_names = list(counter.keys())
    atom_counts = list(counter.values())
    atom_indices = [atom_uni_names.index(atom) for atom in atom_names]
    return atom_uni_names, np.asarray(atom_counts), np.asarray(atom_indices)


def detect_data_start(path):
    """
    Determines where the first line of beamline data is.

    - For '.xyz' files, expects the first column to be a label (e.g., atom type)
      and the last three columns to be floats (X, Y, Z coordinates).
    - For other files, assumes the first two columns are floats.

    Used in dataloader.py and calculator.py.
    """
    ext = os.path.splitext(path)[1].lower()
    is_xyz_format = ext == '.xyz'

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()

            if is_xyz_format:
                if len(parts) >= 4:
                    try:
                        float(parts[-3])
                        float(parts[-2])
                        float(parts[-1])
                        return i
                    except ValueError:
                        continue
            else:
                if len(parts) >= 2:
                    try:
                        float(parts[0])
                        float(parts[1])
                        return i
                    except ValueError:
                        continue

    return 0  # Default if no valid data lines are found


def extract_data(path):
    """
    Attempts to extract experimental beamline data from a given path.
    """
    try:
        start_line = detect_data_start(path)
        data = np.loadtxt(path, skiprows=start_line)
        return data[:, 0], data[:, 1]
    except Exception as e:
        print(f"[Error] Failed to load or parse file '{path}': {e}")
        return


def load_atom_name_positions(file_path):
    """
    Loads atom names and positions from a file, starting from the detected data start.

    Parameters
    ==========
    file_path: str
        Path to the text or .xyz file.

    Returns
    =======
    tuple:
        - atom_names: List[str] – list of atom names
        - atom_positions: np.ndarray – array of atomic positions with shape (n_atoms, 3)
    """
    start_line = detect_data_start(file_path)

    with open(file_path, 'r') as f:
        lines = f.readlines()[start_line:]

    atom_names = []
    atom_positions = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 4:
            continue  # Skip incomplete lines
        atom_names.append(parts[0])
        try:
            atom_positions.append([float(x) for x in parts[-3:]])
        except ValueError:
            continue  # Skip lines with non-numeric coordinates

    atom_positions = np.array(atom_positions)
    return atom_names, atom_positions


def get_q_range(qmin, qmax, len_data):
    return np.linspace(qmin, qmax, len_data, endpoint=False)


def trim_and_pad(x, y, qmin, qmax, q_step=0.01):
    """
    Trims x and y to the range [qmin, qmax] and zero-pads if needed.
    Returns new x and y arrays on a uniform grid with spacing q_step.

    Parameters:
        x (np.ndarray): Input x-axis data.
        y (np.ndarray): Input y-axis data.
        qmin (float): Lower bound for trimming/padding.
        qmax (float): Upper bound for trimming/padding.
        q_step (float): Spacing between points in the output x array.

    Returns:
        (x_new, y_new): Trimmed and padded arrays with uniform spacing.
    """
    x = np.array(x)
    y = np.array(y)

    # Create uniform x array from qmin to qmax
    x_new = np.arange(qmin, qmax + q_step / 2, q_step)

    # Interpolate y onto the new x grid, filling outside range with 0
    mask = (x >= qmin) & (x <= qmax)
    y_new = np.interp(x_new, x[mask], y[mask], left=0, right=0)

    return x_new, y_new


def trim_data_exact(x, y, qmin, qmax):
    """
        ################# 설명 부분 정리해야 됨   12/12/2025
    원본 데이터의 값(Value)과 간격(Step)을 전혀 건드리지 않고,
    오직 qmin ~ qmax 사이의 데이터만 그대로 추출합니다.

    Trims x and y to the range [qmin, qmax] and zero-pads if needed.
    Returns new x and y arrays on a uniform grid with spacing q_step.

    Parameters:
        x (np.ndarray): Input x-axis data.
        y (np.ndarray): Input y-axis data.
        qmin (float): Lower bound for trimming/padding.
        qmax (float): Upper bound for trimming/padding.
        q_step (float): Spacing between points in the output x array.

    Returns:
        (x_new, y_new): Trimmed and padded arrays with uniform spacing.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    # 1. 마스킹: 원본 x축에서 원하는 범위 내에 있는 데이터의 위치(Index)만 찾음
    mask = (x >= qmin) & (x <= qmax)

    # 2. 슬라이싱: 해당 위치의 x, y 값만 원본 그대로 가져옴
    # (새로운 grid를 만들거나 interp를 하지 않음)
    return x[mask], y[mask]





