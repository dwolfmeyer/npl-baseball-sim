"""Where hits go and how far runners get, measured from Retrosheet 2023 to 2025.

Regenerate with `python -m npl_sim.calibration.retrosheet <season plays.csv files>`.
FIELD_SHARES[(result, ball type, batter side)] is where a single or double was hit
(IF, LF, CF, RF). ADVANCE[situation][(outs, field)] is (P(runner takes the extra base),
P(thrown out trying)) for a runner on second on a single (extra base: scores), on first on a
single with second open (extra: reaches third or scores) and on first on a double (extra:
scores). Thin cells are shrunk toward their outs row.
"""

FIELD_SHARES = {
    ("1B", "FB", "L"): {"CF": 0.351, "IF": 0.064, "LF": 0.463, "RF": 0.121},
    ("1B", "FB", "R"): {"CF": 0.425, "IF": 0.057, "LF": 0.241, "RF": 0.276},
    ("1B", "GB", "L"): {"CF": 0.0, "IF": 1.0, "LF": 0.0, "RF": 0.0},
    ("1B", "GB", "R"): {"CF": 0.0, "IF": 1.0, "LF": 0.0, "RF": 0.0},
    ("1B", "LD", "L"): {"CF": 0.392, "IF": 0.032, "LF": 0.302, "RF": 0.274},
    ("1B", "LD", "R"): {"CF": 0.391, "IF": 0.033, "LF": 0.414, "RF": 0.162},
    ("1B", "PU", "L"): {"CF": 0.078, "IF": 0.661, "LF": 0.235, "RF": 0.026},
    ("1B", "PU", "R"): {"CF": 0.08, "IF": 0.724, "LF": 0.037, "RF": 0.16},
    ("2B", "FB", "L"): {"CF": 0.298, "IF": 0.026, "LF": 0.47, "RF": 0.206},
    ("2B", "FB", "R"): {"CF": 0.299, "IF": 0.013, "LF": 0.454, "RF": 0.233},
    ("2B", "GB", "L"): {"CF": 0.0, "IF": 1.0, "LF": 0.0, "RF": 0.0},
    ("2B", "GB", "R"): {"CF": 0.0, "IF": 1.0, "LF": 0.0, "RF": 0.0},
    ("2B", "LD", "L"): {"CF": 0.258, "IF": 0.018, "LF": 0.309, "RF": 0.416},
    ("2B", "LD", "R"): {"CF": 0.198, "IF": 0.015, "LF": 0.655, "RF": 0.131},
    ("2B", "PU", "L"): {"CF": 0.103, "IF": 0.621, "LF": 0.276, "RF": 0.0},
    ("2B", "PU", "R"): {"CF": 0.167, "IF": 0.567, "LF": 0.033, "RF": 0.233},
}

ADVANCE = {
    "double_1st": {
        (0, "CF"): (0.33, 0.029),
        (0, "IF"): (0.257, 0.015),
        (0, "LF"): (0.301, 0.018),
        (0, "RF"): (0.194, 0.017),
        (1, "CF"): (0.437, 0.029),
        (1, "IF"): (0.318, 0.015),
        (1, "LF"): (0.333, 0.023),
        (1, "RF"): (0.214, 0.022),
        (2, "CF"): (0.69, 0.038),
        (2, "IF"): (0.398, 0.038),
        (2, "LF"): (0.553, 0.038),
        (2, "RF"): (0.454, 0.045),
    },
    "single_1st": {
        (0, "CF"): (0.265, 0.006),
        (0, "IF"): (0.284, 0.01),
        (0, "LF"): (0.166, 0.018),
        (0, "RF"): (0.392, 0.006),
        (1, "CF"): (0.296, 0.01),
        (1, "IF"): (0.305, 0.012),
        (1, "LF"): (0.166, 0.012),
        (1, "RF"): (0.41, 0.007),
        (2, "CF"): (0.477, 0.013),
        (2, "IF"): (0.342, 0.013),
        (2, "LF"): (0.273, 0.017),
        (2, "RF"): (0.513, 0.02),
    },
    "single_2nd": {
        (0, "CF"): (0.397, 0.021),
        (0, "IF"): (0.461, 0.013),
        (0, "LF"): (0.393, 0.021),
        (0, "RF"): (0.333, 0.019),
        (1, "CF"): (0.565, 0.024),
        (1, "IF"): (0.545, 0.019),
        (1, "LF"): (0.551, 0.031),
        (1, "RF"): (0.422, 0.026),
        (2, "CF"): (0.934, 0.026),
        (2, "IF"): (0.683, 0.033),
        (2, "LF"): (0.861, 0.051),
        (2, "RF"): (0.825, 0.061),
    },
}


def odds(situation: str, outs: int, field: str) -> tuple[float, float]:
    """(take, safe) for `Game._send`: how often the runner ends up on the extra base, and how
    often a runner who goes is safe."""
    extra, out = ADVANCE[situation][(min(outs, 2), field)]
    return extra, extra / (extra + out)


def pick_field(rng, result: str, bip: str, side: str) -> str:
    """Where a single or double was hit: IF, LF, CF or RF."""
    shares = FIELD_SHARES[(result, bip, side)]
    roll = rng.random()
    for field, share in shares.items():
        roll -= share
        if roll < 0:
            return field
    return "IF"
