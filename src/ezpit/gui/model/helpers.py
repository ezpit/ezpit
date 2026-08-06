from collections import Counter
import numpy as np
import os


# helpers gathered from EZPIT

def parse_composition(composition):
    """
    Parse a composition string from the Control Panel and turns it into a dictionary.

    Both the spaced and the compact writing style are accepted, and a quantity
    of 1 may be omitted. All of the following give {'Co': 38, 'O': 119, 'P': 1}:

        'Co 38 O 119 P 1'     (spaced, explicit 1)
        'Co 38 O 119 P'       (spaced, omitted 1)
        'Co38O119P1'          (compact, explicit 1)
        'Co38O119P'           (compact, omitted 1)

    Further examples:
        'Si 1 O 2' --> {'Si': 1, 'O': 2}
        'SiO2'     --> {'Si': 1, 'O': 2}
        'CoO119P'  --> {'Co': 1, 'O': 119, 'P': 1}
        'Co38OP'   --> {'Co': 38, 'O': 1, 'P': 1}

    Fractional amounts are also accepted, so a composition may be written the
    way it usually appears in a paper. Scaling every element by the same factor
    describes the same material and gives identical form-factor averages:

        'Li0.2Co0.36Mn0.37Ni0.07'  ==  'Li20Co36Mn37Ni7'

    Note that the factor must be applied to *every* element. Scaling only one
    of them ('Li20Co36Mn370Ni7') describes a different material.

    Rules:
      - An element symbol is one uppercase letter optionally followed by one
        lowercase letter (H, O, Si, Co, Mn, ...). Correct capitalisation is
        required, exactly as elsewhere in EZPDF.
      - A quantity is a positive number, integer or decimal; if omitted it
        defaults to 1. Whole numbers are returned as int, so ordinary
        compositions behave exactly as before.
      - Whitespace anywhere is ignored.
      - A repeated element is summed ('CoOCo' --> {'Co': 2, 'O': 1}).
    """

    if composition is None or composition.strip() == "":
        raise ValueError("The composition field must not be empty")

    # Drop all whitespace so 'Co 38 O 119' and 'Co38O119' parse identically.
    compact = "".join(ch for ch in composition if not ch.isspace())

    composition_dict = Counter()
    i = 0
    length = len(compact)
    while i < length:
        ch = compact[i]

        # An element symbol must start with an uppercase letter.
        if not ch.isupper():
            raise ValueError(
                f'Cannot read "{ch}" — expected an element symbol starting '
                f'with a capital letter (e.g. Co 38 O 119 P 1 or Co38O119P)'
            )

        # Read the symbol: uppercase letter + optional one lowercase letter.
        element = ch
        i += 1
        if i < length and compact[i].islower():
            element += compact[i]
            i += 1

        # Read the quantity: digits with at most one decimal point, or nothing
        # (in which case it defaults to 1).
        digits = ""
        seen_dot = False
        while i < length and (compact[i].isdigit() or compact[i] == "."):
            if compact[i] == ".":
                if seen_dot:
                    raise ValueError(
                        f"'{element}': count has more than one decimal point"
                    )
                seen_dot = True
            digits += compact[i]
            i += 1

        if digits == "":
            quantity = 1
        else:
            try:
                quantity = float(digits)
            except ValueError:
                raise ValueError(
                    f"'{element}': '{digits}' is not a valid number"
                )
            if quantity <= 0:
                raise ValueError(
                    f"'{element}': count must be a positive number (got {digits})"
                )
            # Keep whole numbers as int so that existing integer-based code
            # (and the preview text) is unchanged for ordinary compositions.
            if float(quantity).is_integer():
                quantity = int(quantity)

        composition_dict[element] += quantity

    if not composition_dict:
        raise ValueError(
            f'Invalid composition string "{composition}": no element found'
        )

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

    # Reuse parse_composition so the preview accepts exactly the same styles as
    # the actual processing does (spaced or compact, with 1 optionally omitted).
    # Its error text is passed straight through so the user is told precisely
    # what is wrong rather than being given a generic hint.
    try:
        parsed_counter = parse_composition(text)
    except ValueError as exc:
        return {'ok': False, 'message': str(exc), 'parsed': {}}

    valid = _load_valid_elements()  # may be None -> skip existence check
    parsed = {}
    errors = []
    for el, n in parsed_counter.items():
        if valid is not None and el not in valid:
            errors.append(f"'{el}' is not a known element")
            continue
        parsed[el] = parsed.get(el, 0) + n

    if errors:
        return {'ok': False, 'message': "  •  ".join(errors), 'parsed': parsed}

    parts = [f"{el} ({ELEMENT_FULL_NAMES.get(el, '?')}) {n:g}"
             for el, n in parsed.items()]
    total = sum(parsed.values())
    is_fractional = any(not float(n).is_integer() for n in parsed.values())
    if is_fractional:
        # Fractional composition: the total is a ratio, not an atom count.
        message = " ,   ".join(parts) + f"      (total {total:g})"
    else:
        message = " ,   ".join(parts) + f"      ({int(total)} atoms)"
    return {'ok': True, 'message': message, 'parsed': parsed}


def convert_atom_names(composition: dict) -> list[str]:
    """
    if composition = {'Co': 3, 'O': 4, 'P': 1},
    atom_names = losa.convert_atom_names(composition)
    print('atom_names = ', atom_names)
    --> atom_names = ['Co', 'Co', 'Co', 'O', 'O', 'O', 'O', 'P']

    Only whole-number compositions can be expanded this way. For a composition
    that may contain fractional amounts, use composition_weights() instead.
    """
    for el, count in composition.items():
        if not float(count).is_integer():
            raise ValueError(
                f"'{el}': fractional amount {count} cannot be expanded into "
                f"individual atoms. Use composition_weights() instead, or "
                f"scale every element by the same factor "
                f"(e.g. Li0.2Co0.36 -> Li20Co36)."
            )
    return [el for el, count in composition.items() for _ in range(int(count))]


def composition_weights(composition: dict):
    """
    Turn a composition dictionary into unique element names and their amounts.

    This is the fraction-safe counterpart of convert_atom_names() +
    group_atoms(). Rather than listing one entry per atom, it keeps one entry
    per unique element together with how much of it is present, which is all
    the form-factor averages actually need:

        <f>   = sum_k(c_k * f_k)   / sum_k(c_k)
        <f^2> = sum_k(c_k * f_k^2) / sum_k(c_k)

    Because both averages are normalised by the total, scaling every element by
    the same factor leaves the result unchanged. These all give identical
    averages:

        {'Li': 0.2, 'Co': 0.36, 'Mn': 0.37, 'Ni': 0.07}
        {'Li': 20,  'Co': 36,   'Mn': 37,   'Ni': 7}
        {'Li': 200, 'Co': 360,  'Mn': 370,  'Ni': 70}

    Returns
    -------
    (list[str], numpy.ndarray)
        Unique element names, and their amounts as floats in the same order.
    """
    if not composition:
        raise ValueError("The composition must not be empty")

    names = list(composition.keys())
    weights = np.asarray([float(composition[el]) for el in names], dtype=float)

    if np.any(weights <= 0):
        raise ValueError("All composition amounts must be positive")

    return names, weights


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

    # Read as latin-1 so any byte decodes without error. Reduction tools often
    # write non-UTF-8 characters in the header (e.g. the 'µ' in 'µm'), which
    # would otherwise crash a strict UTF-8 read.
    with open(path, 'r', encoding='latin-1') as f:
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
    Attempts to extract experimental beamline data (two columns) from a path.

    Automatically skips any header and handles the file variety produced by
    common reduction tools: plain two-column .dat files, pyFAI / Fit2d '.xy'
    files with '#' headers, JSON-style '#' headers, Windows or Unix line
    endings, non-UTF-8 header bytes (e.g. 'µm'), and whitespace- or
    comma-separated columns.
    """
    try:
        # latin-1 decodes every byte without error; we only need the numbers.
        with open(path, 'r', encoding='latin-1') as f:
            lines = f.readlines()

        x_vals, y_vals = [], []
        for line in lines:
            s = line.strip()
            if not s or s[0] in ('#', '!', ';', '%'):
                continue
            parts = s.replace(',', ' ').split()
            if len(parts) < 2:
                continue
            try:
                a = float(parts[0])
                b = float(parts[1])
            except ValueError:
                continue
            x_vals.append(a)
            y_vals.append(b)

        if len(x_vals) < 2:
            raise ValueError("No two-column numeric data found.")

        return np.asarray(x_vals, dtype=float), np.asarray(y_vals, dtype=float)
    except Exception as e:
        print(f"[Error] Failed to load or parse file '{path}': {e}")
        return


_AFF_VALID_SYMBOLS = None


def _get_aff_valid_symbols():
    """Accepted species symbols (elements and ions) from the aff form-factor
    table, e.g. ['H', ..., 'Fe2+', 'O2-', ...]. Cached after first read.
    Returns None if the table can't be loaded."""
    global _AFF_VALID_SYMBOLS
    if _AFF_VALID_SYMBOLS is None:
        try:
            from ezpit.elem_tables import AFF_ELEMENTS
            _AFF_VALID_SYMBOLS = [str(s).strip() for s in AFF_ELEMENTS]
        except Exception:
            _AFF_VALID_SYMBOLS = []
    return _AFF_VALID_SYMBOLS or None


def _load_atom_name_positions_legacy(file_path):
    """Positional fallback parser: skips header via detect_data_start and takes
    the last three columns as x, y, z. Used only when the form-factor table is
    unavailable."""
    start_line = detect_data_start(file_path)
    with open(file_path, 'r') as f:
        lines = f.readlines()[start_line:]
    atom_names = []
    atom_positions = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        atom_names.append(parts[0])
        try:
            atom_positions.append([float(x) for x in parts[-3:]])
        except ValueError:
            continue
    return atom_names, np.array(atom_positions)


def load_atom_name_positions(file_path, valid_symbols=None):
    """
    Load atom names and (x, y, z) positions from an .xyz file, using the atomic
    form-factor table as the reference for what counts as an atom.

    A line is treated as an atom only when:
      (1) it has at least 4 whitespace-separated tokens,
      (2) its first token is an accepted species symbol (an element OR ion
          present in the aff form-factor table), and
      (3) the three tokens immediately after the symbol parse as floats.

    The stored symbol is normalised to the canonical table entry (exact match
    first, so ions like 'Fe2+'/'O2-' are kept; otherwise the element part is
    case-normalised, e.g. 'FE' -> 'Fe'). This guarantees each atom maps to the
    correct form factor. Coordinates are read from the three columns right after
    the symbol, so trailing columns (charge, force, ...) are ignored.

    Header handling is automatic and works whether the file has many header
    lines, a single comment line, or none at all:
      * Standard .xyz: if an atom-count line (a line holding a single positive
        integer N) is found and is followed by a comment line and then N clean
        atom lines, exactly those N atoms are read. The comment line is skipped
        unconditionally, so it is never mistaken for an atom even if it happens
        to look like one.
      * Otherwise (headerless or non-standard files): every atom-like line in
        the file is taken and everything else is skipped.

    Parameters
    ==========
    file_path : str
        Path to the .xyz file.
    valid_symbols : iterable of str, optional
        Accepted species symbols. When omitted, they are taken from the aff
        form-factor table automatically.

    Returns
    =======
    tuple:
        - atom_names: List[str] – canonical species symbols
        - atom_positions: np.ndarray – (n_atoms, 3) float array
    """
    if valid_symbols is None:
        valid_symbols = _get_aff_valid_symbols()

    # If the form-factor table can't be read, fall back to positional parsing
    # so the app still works rather than failing outright.
    if not valid_symbols:
        return _load_atom_name_positions_legacy(file_path)

    symbol_set = set(valid_symbols)

    def normalize_symbol(tok):
        # Exact match first (keeps ions as written in the table), then a
        # case-normalised element part (e.g. 'FE' -> 'Fe', 'fe2+' -> 'Fe2+').
        if tok in symbol_set:
            return tok
        cap = tok[:1].upper() + tok[1:].lower()
        if cap in symbol_set:
            return cap
        return None

    def atom_of(line):
        # Return (symbol, [x, y, z]) if the line is a clean atom line, else None.
        parts = line.split()
        if len(parts) < 4:
            return None
        symbol = normalize_symbol(parts[0])
        if symbol is None:
            return None
        try:
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
        except ValueError:
            return None
        return symbol, xyz

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # --- Preferred path: standard .xyz with a count line ---------------------
    # The line after the count is the comment BY SPEC, so it is skipped
    # unconditionally (never read as an atom, even if it looks like one).
    n_lines = len(lines)
    for i in range(n_lines):
        toks = lines[i].split()
        if len(toks) != 1 or not toks[0].isdigit():
            continue
        n_atoms = int(toks[0])
        if n_atoms <= 0:
            continue
        block = []
        k = i + 2  # skip the count line (i) and the comment line (i + 1)
        while k < n_lines and len(block) < n_atoms:
            if lines[k].strip() == "":   # tolerate blank lines within the block
                k += 1
                continue
            a = atom_of(lines[k])
            if a is None:                # not a clean atom where one was expected
                break
            block.append(a)
            k += 1
        if len(block) == n_atoms:
            names = [b[0] for b in block]
            pos = [b[1] for b in block]
            return names, np.array(pos, dtype=float)
        # Count line didn't lead to a clean N-atom block; keep looking, then
        # fall through to the content scan below.

    # --- Fallback: headerless / non-standard files ---------------------------
    # Take every atom-like line and skip everything else (any number of header
    # or footer lines, in any style).
    atom_names = []
    atom_positions = []
    for line in lines:
        a = atom_of(line)
        if a is None:
            continue
        atom_names.append(a[0])
        atom_positions.append(a[1])

    if not atom_positions:
        raise ValueError(
            "No atom coordinates found in '{0}'. Expected lines of the form "
            "'Element x y z' where the symbol is present in the atomic "
            "form-factor table.".format(file_path)
        )

    return atom_names, np.array(atom_positions, dtype=float)


def composition_string_from_xyz(file_path):
    """Build a compact composition string (e.g. 'Co38O119P1') from an .xyz file.

    Counts each unique element in the file's atom list. Returns None if the
    file cannot be read or contains no recognisable atoms. The result parses
    back cleanly with parse_composition().
    """
    try:
        atom_names, _ = load_atom_name_positions(file_path)
    except Exception:
        return None

    if not atom_names:
        return None

    unique_names, counts, _ = group_atoms(atom_names)
    parts = []
    for name, count in zip(unique_names, counts):
        parts.append(f"{name}{int(count)}")
    return "".join(parts) if parts else None


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



