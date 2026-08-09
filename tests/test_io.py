import numpy as np
import pytest

from ezpit.io import composition_weights, convert_atom_names, group_atoms, load_atom_name_positions, parse_composition


@pytest.mark.parametrize(
    ("composition", "expected"),
    [
        # Compact string, explicit counts.
        ("Co38O119P1", {"Co": 38, "O": 119, "P": 1}),
        # Trailing count of 1 may be omitted.
        ("Co38O119P", {"Co": 38, "O": 119, "P": 1}),
        # Spaced style parses identically to the compact style.
        ("Co 38 O 119 P 1", {"Co": 38, "O": 119, "P": 1}),
        # Count of 1 omitted throughout.
        ("SiO2", {"Si": 1, "O": 2}),
        # Single element, no count.
        ("H", {"H": 1}),
        # Repeated elements are summed.
        ("CoOCo", {"Co": 2, "O": 1}),
        # Fractional amounts are kept as-is (float).
        ("Li0.2Co0.36Mn0.37Ni0.07", {"Li": 0.2, "Co": 0.36, "Mn": 0.37, "Ni": 0.07}),
        # Whitespace anywhere is ignored.
        ("  Fe 2  O 3 ", {"Fe": 2, "O": 3}),
        # A dict input is returned unchanged.
        ({"Co": 2, "O": 1}, {"Co": 2, "O": 1}),
    ],
)
def test_parse_composition_valid(composition: str | dict[str, float], expected: dict[str, float]):
    result = parse_composition(composition)
    assert dict(result) == expected


def test_parse_composition_keeps_whole_numbers_as_int():
    result = parse_composition("Co2O3")
    assert isinstance(result["Co"], int)
    assert isinstance(result["O"], int)


def test_parse_composition_keeps_fractions_as_float():
    result = parse_composition("Li0.2Co0.8")
    assert isinstance(result["Li"], float)
    assert result["Li"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "composition",
    [
        "",  # empty string
        "   ",  # whitespace only
        None,  # None
    ],
)
def test_parse_composition_empty_raises(composition: str | None):
    with pytest.raises(ValueError, match="must not be empty"):
        parse_composition(composition)


@pytest.mark.parametrize(
    "composition",
    [
        "Fe2+",  # cation
        "O2-",  # anion
        "Cl1-",  # anion with explicit charge
    ],
)
def test_parse_composition_rejects_ions(composition: str):
    with pytest.raises(ValueError, match="ion"):
        parse_composition(composition)


@pytest.mark.parametrize(
    "composition",
    [
        "38Co",  # starts with a digit, not an element symbol
        "co38",  # lowercase start is not a valid symbol
        "Co1.2.3",  # more than one decimal point in a count
    ],
)
def test_parse_composition_invalid_raises(composition: str):
    with pytest.raises(ValueError):
        parse_composition(composition)


@pytest.mark.parametrize(
    ("composition", "expected_names", "expected_weights"),
    [
        # Compact integer string.
        ("Co38O119P1", ["Co", "O", "P"], [38.0, 119.0, 1.0]),
        # Count of 1 omitted.
        ("SiO2", ["Si", "O"], [1.0, 2.0]),
        # Fractional amounts are kept as floats.
        ("Li0.2Co0.8", ["Li", "Co"], [0.2, 0.8]),
        # Repeated elements are summed.
        ("CoOCo", ["Co", "O"], [2.0, 1.0]),
        # Dict input.
        ({"Fe": 2, "O": 3}, ["Fe", "O"], [2.0, 3.0]),
    ],
)
def test_composition_weights_valid(
    composition: str | dict[str, float],
    expected_names: list[str],
    expected_weights: list[float],
):
    names, weights = composition_weights(composition)
    assert names == expected_names
    assert isinstance(weights, np.ndarray)
    assert weights.dtype == np.float64
    np.testing.assert_allclose(weights, expected_weights)


def test_composition_weights_order_matches_names():
    names, weights = composition_weights("Co2O1P3")
    assert names == ["Co", "O", "P"]
    np.testing.assert_allclose(weights, [2.0, 1.0, 3.0])


@pytest.mark.parametrize("composition", ["", "   ", {}])
def test_composition_weights_empty_raises(composition: str | dict[str, float]):
    with pytest.raises(ValueError, match="must not be empty"):
        composition_weights(composition)


@pytest.mark.parametrize(
    "composition",
    [
        {"Co": 0},  # zero amount
        {"Co": -1},  # negative amount
    ],
)
def test_composition_weights_nonpositive_raises(composition: dict[str, float]):
    with pytest.raises(ValueError, match="positive"):
        composition_weights(composition)


@pytest.mark.parametrize(
    ("composition", "expected"),
    [
        # Dict input, one entry per atom in dict order.
        ({"Co": 3, "O": 4, "P": 1}, ["Co", "Co", "Co", "O", "O", "O", "O", "P"]),
        # Compact string.
        ("Co3O4P1", ["Co", "Co", "Co", "O", "O", "O", "O", "P"]),
        # Spaced string parses identically.
        ("Co 3 O 4 P", ["Co", "Co", "Co", "O", "O", "O", "O", "P"]),
        # Count of 1 omitted.
        ("SiO2", ["Si", "O", "O"]),
        # Single atom.
        ("H", ["H"]),
    ],
)
def test_convert_atom_names_valid(composition: str | dict[str, float], expected: list[str]):
    assert convert_atom_names(composition) == expected


@pytest.mark.parametrize(
    "composition",
    [
        "Li0.2Co0.8",  # fractional string
        {"Li": 0.2, "Co": 0.8},  # fractional dict
    ],
)
def test_convert_atom_names_fractional_raises(composition: str | dict[str, float]):
    with pytest.raises(ValueError, match="fractional"):
        convert_atom_names(composition)


@pytest.mark.parametrize(
    ("atom_names", "expected_names", "expected_counts", "expected_indices"),
    [
        # Grouped input: consecutive repeats.
        (
            ["Co", "Co", "Co", "O", "O", "O", "O", "P"],
            ["Co", "O", "P"],
            [3, 4, 1],
            [0, 0, 0, 1, 1, 1, 1, 2],
        ),
        # Interleaved input: unique-name order follows first appearance.
        (
            ["Co", "O", "Co", "P", "O"],
            ["Co", "O", "P"],
            [2, 2, 1],
            [0, 1, 0, 2, 1],
        ),
        # Single element repeated.
        (["H", "H", "H"], ["H"], [3], [0, 0, 0]),
        # Single atom.
        (["Fe"], ["Fe"], [1], [0]),
    ],
)
def test_group_atoms_valid(
    atom_names: list[str],
    expected_names: list[str],
    expected_counts: list[int],
    expected_indices: list[int],
):
    names, counts, indices = group_atoms(atom_names)

    # Unique names preserve first-appearance order.
    assert names == expected_names

    # Counts and indices are numpy integer arrays.
    assert isinstance(counts, np.ndarray)
    assert isinstance(indices, np.ndarray)
    assert np.issubdtype(counts.dtype, np.integer)
    assert np.issubdtype(indices.dtype, np.integer)

    np.testing.assert_array_equal(counts, np.asarray(expected_counts))
    np.testing.assert_array_equal(indices, np.asarray(expected_indices))


def test_group_atoms_counts_sum_matches_length():
    atom_names = ["Co", "O", "Co", "P", "O", "O"]
    _, counts, indices = group_atoms(atom_names)

    # Total count equals the number of atoms passed in.
    assert int(counts.sum()) == len(atom_names)
    # One index per atom, each pointing into the unique-name list.
    assert len(indices) == len(atom_names)


def test_group_atoms_indices_map_back_to_names():
    atom_names = ["Co", "O", "Co", "P", "O"]
    names, _, indices = group_atoms(atom_names)

    # Reconstructing names from indices reproduces the original input.
    assert [names[i] for i in indices] == atom_names


# ----------------------------------------------------------------------------------
# load_atom_name_positions
# ----------------------------------------------------------------------------------
# Accepted-symbol table used by the tests (kept small and explicit).
VALID_SYMBOLS = ["Co", "O", "P", "Si", "Fe", "Fe2+", "O2-"]


@pytest.mark.parametrize(
    ("contents", "expected_names", "expected_positions"),
    [
        # Standard .xyz: count line + comment line + atom lines (both skipped/read).
        (
            "3\ncomment line\nCo 0.0 0.0 0.0\nO 1.0 2.0 3.0\nP -1.5 0.5 2.5\n",
            ["Co", "O", "P"],
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.5, 0.5, 2.5]],
        ),
        # No header at all — atom lines are still detected.
        (
            "Si 0.0 0.0 0.0\nO 1.0 1.0 1.0\n",
            ["Si", "O"],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        ),
        # Extra columns after x, y, z (charge/force) are ignored.
        (
            "Co 0.1 0.2 0.3 0.5 1.0\n",
            ["Co"],
            [[0.1, 0.2, 0.3]],
        ),
        # Case-normalised element part ('FE' -> 'Fe', 'co' -> 'Co').
        (
            "FE 0.0 0.0 0.0\nco 1.0 1.0 1.0\n",
            ["Fe", "Co"],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        ),
        # Ions in the table are matched exactly.
        (
            "Fe2+ 0.0 0.0 0.0\nO2- 1.0 1.0 1.0\n",
            ["Fe2+", "O2-"],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        ),
        # Blank lines and unknown-symbol lines are skipped.
        (
            "\nXx 0.0 0.0 0.0\nCo 1.0 2.0 3.0\n\n",
            ["Co"],
            [[1.0, 2.0, 3.0]],
        ),
    ],
)
def test_load_atom_name_positions_valid(
    tmp_path,
    contents: str,
    expected_names: list[str],
    expected_positions: list[list[float]],
):
    xyz_file = tmp_path / "structure.xyz"
    xyz_file.write_text(contents)

    names, positions = load_atom_name_positions(xyz_file, VALID_SYMBOLS)

    assert names == expected_names
    assert isinstance(positions, np.ndarray)
    assert positions.dtype == np.float64
    assert positions.shape == (len(expected_names), 3)
    np.testing.assert_allclose(positions, np.asarray(expected_positions))


def test_load_atom_name_positions_accepts_str_path(tmp_path):
    # A plain string path is converted to Path internally.
    xyz_file = tmp_path / "structure.xyz"
    xyz_file.write_text("Co 0.0 0.0 0.0\n")

    names, positions = load_atom_name_positions(str(xyz_file), VALID_SYMBOLS)

    assert names == ["Co"]
    np.testing.assert_allclose(positions, np.asarray([[0.0, 0.0, 0.0]]))


def test_load_atom_name_positions_no_atoms_raises(tmp_path):
    # A file with only header/unknown lines yields no atoms.
    xyz_file = tmp_path / "empty.xyz"
    xyz_file.write_text("2\njust a comment\nXx 0.0 0.0 0.0\n")

    with pytest.raises(ValueError, match="No atom coordinates found"):
        load_atom_name_positions(xyz_file, VALID_SYMBOLS)
