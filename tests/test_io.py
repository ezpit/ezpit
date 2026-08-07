import pytest
from ezpit.io import parse_composition


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
def test_parse_composition_valid(composition, expected):
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
def test_parse_composition_empty_raises(composition):
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
def test_parse_composition_rejects_ions(composition):
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
def test_parse_composition_invalid_raises(composition):
    with pytest.raises(ValueError):
        parse_composition(composition)
