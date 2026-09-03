# -*- coding: utf-8 -*-
"""
spot.utils.data_io — loading of packaged / user data files
===========================================================

All non-code data files of spot (abundance tables, line lists, preset
atmosphere models) live under ``spot/data`` and are shipped with the
package through ``setup.py``'s ``package_data``.  They are located at
run time with :func:`resource_path`, which tries modern
``importlib.resources`` first and falls back to ``pkg_resources``
(setuptools) for older Python versions — the two standard ways to
access files that travel inside an installed Python package.

User-provided files (a path to a custom abundance table, line list or
model file) are always honoured: pass a path string instead of a preset
name.
"""

import csv
import os
import re

__all__ = [
    "resource_path",
    "list_presets",
    "load_abundance",
    "load_lines",
    "load_atmosphere",
    "resolve_preset",
    "search_lines",
]

# ---------------------------------------------------------------------------
# resource access
# ---------------------------------------------------------------------------
_PACKAGE = "spot"
_DATA_DIR = "data"


def resource_path(*parts):
    """Absolute path of a data file shipped inside the spot package."""
    rel = os.path.join(_DATA_DIR, *parts)
    try:  # Python >= 3.9 (importlib.resources)
        from importlib.resources import files
        return str(files(_PACKAGE).joinpath(rel))
    except Exception:  # pragma: no cover - older Pythons / zipped installs
        import pkg_resources
        return pkg_resources.resource_filename(_PACKAGE, rel)


def list_presets(kind):
    """
    List available packaged preset names for a given kind.

    Parameters
    ----------
    kind : str
        'models' (atmosphere presets), 'lines' or 'abundance'.
    """
    if kind == "models":
        d = resource_path("models")
        names = []
        for fname in sorted(os.listdir(d)):
            if fname.endswith(".csv"):
                names.append(os.path.splitext(fname)[0].lower())
        return names
    if kind == "lines":
        return ["default"]
    if kind == "abundance":
        return ["thevenin"]
    raise ValueError(f"unknown preset kind {kind!r}")


def resolve_preset(kind, name):
    """
    Return the file path of a preset, or the path itself if ``name`` is
    already a path-like string that exists on disk.

    If ``name`` does not match any preset and is not an existing file,
    a helpful error listing the available presets is raised.
    """
    if name is None:
        return None
    if os.path.isfile(name):
        return name
    candidates = []
    if kind == "models":
        base = os.path.join(resource_path("models"), f"{name.lower()}.csv")
        if os.path.isfile(base):
            return base
        candidates = list_presets("models")
    elif kind == "lines":
        if name == "default":
            return resource_path("lines.csv")
        candidates = ["default"]
    elif kind == "abundance":
        if name == "thevenin":
            return resource_path("abundance.csv")
        candidates = ["thevenin"]
    raise FileNotFoundError(
        f"preset {name!r} for {kind!r} not found. Available presets: "
        f"{candidates} — or pass a path to an existing file.")


# ---------------------------------------------------------------------------
# abundance
# ---------------------------------------------------------------------------
def load_abundance(source="thevenin"):
    """
    Load an abundance table.

    Parameters
    ----------
    source : str, optional
        Preset name ('thevenin', default) or a path to a CSV file with
        columns ``atomic_number, abundance_log12`` (the 12+log10(N/N_H)
        scale, the abundance convention of the reference data).

    Returns
    -------
    dict
        ``{'atomic_number': np.ndarray[int], 'abundance_log12': np.ndarray[float]}``
    """
    import numpy as np

    path = resolve_preset("abundance", source)
    z, abu = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or row[0].startswith("#") or row[0] == "atomic_number":
                continue
            z.append(int(row[0]))
            abu.append(float(row[1]))
    return {"atomic_number": np.asarray(z, dtype=int),
            "abundance_log12": np.asarray(abu, dtype=float)}


# ---------------------------------------------------------------------------
# line list
# ---------------------------------------------------------------------------
_LINE_FIELDS = [
    "index", "atom", "stage", "wavelength", "zeff", "energy_low",
    "loggf", "mult_l", "design_l", "tam_l",
    "mult_u", "design_u", "tam_u", "alfa", "sigma",
]

# one line-list record, e.g.:
#   1=FE 1  6301.5008   1.0  3.654   -0.718   5P 2.0- 5D 2.0  0.243  2.35e-14
_LINE_RECORD_RE = re.compile(
    r"^(\d+)=([A-Z]{1,2})\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+"
    r"(-?[\d.]+)\s+([-+]?[\d.]+)\s+(\d+)([A-Za-z0-9])\s+"
    r"([\d.]+)-\s*(\d+)([A-Za-z0-9])\s+([\d.]+)\s+"
    r"(-?[\d.]+)\s+(-?[\d.]+(?:[eE][-+]?\d+)?)")


def _parse_line_record(line):
    """Parse one line-list record into a dict of the _LINE_FIELDS."""
    line = line.strip()
    m = _LINE_RECORD_RE.match(line)
    if not m:
        raise ValueError(f"cannot parse line record: {line!r}")
    (idx, atom, stage, wl, zeff, energy, loggf,
     mult_l, des_l, tam_l, mult_u, des_u, tam_u, alfa, sigma) = m.groups()
    return {
        "index": int(idx), "atom": atom.strip().upper(), "stage": int(stage),
        "wavelength": float(wl), "zeff": float(zeff),
        "energy_low": float(energy), "loggf": float(loggf),
        "mult_l": int(mult_l), "design_l": des_l.upper(), "tam_l": float(tam_l),
        "mult_u": int(mult_u), "design_u": des_u.upper(), "tam_u": float(tam_u),
        "alfa": float(alfa), "sigma": float(sigma),
    }


def load_lines(source="default"):
    """
    Load a list of spectral lines.

    Parameters
    ----------
    source : str, optional
        Preset name ('default') or a path to a CSV file with the
        columns of :data:`_LINE_FIELDS` (same layout as the converted
        line-list file) or a raw line-list file.

    Returns
    -------
    list of dict
        One dict per line with fields:

        index, atom, stage, wavelength [Angstrom], zeff, energy_low [eV],
        loggf, mult_l, design_l, tam_l, mult_u, design_u, tam_u,
        alfa, sigma   (alfa/sigma: ABO damping parameters; 0 -> Unsöld)
    """
    path = resolve_preset("lines", source)
    with open(path, newline="", encoding="utf-8") as fh:
        head = fh.read(2048)
    if "index" in head.splitlines()[0] if head.splitlines() else "":
        # CSV format (converted preset)
        lines = []
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rec = {k: row[k] for k in _LINE_FIELDS}
                rec["index"] = int(rec["index"])
                rec["stage"] = int(rec["stage"])
                for k in ("wavelength", "zeff", "energy_low", "loggf",
                          "tam_l", "tam_u", "alfa", "sigma"):
                    rec[k] = float(rec[k])
                rec["mult_l"] = int(rec["mult_l"])
                rec["mult_u"] = int(rec["mult_u"])
                lines.append(rec)
        return lines
    # raw line-list format, e.g.:
    #   1=FE 1  6301.5008   1.0  3.654   -0.718   5P 2.0- 5D 2.0  0.243  2.35e-14
    return _parse_lines_file(path)


def _parse_lines_file(path):
    """Parse a raw line-list file (records like ``1=FE 1 6301.5 ...``)."""
    records = []
    with open(path, "r", encoding="latin-1") as fh:
        for line in fh:
            line = line.strip()
            if not line or "=" not in line:
                continue
            try:
                records.append(_parse_line_record(line))
            except ValueError as exc:
                print(f"  [warn] skipping line record: {exc}")
    return records


# ---------------------------------------------------------------------------
# atmosphere models
# ---------------------------------------------------------------------------
_ATMOS_FIELDS = ["ltau", "T", "Pe", "vmic", "B", "vlos", "gamma", "phi",
                 "Pg", "z", "rho"]


def load_atmosphere(name="hot11", ltau=None, interpolation="linear"):
    """
    Load a preset (or user-supplied) atmosphere model.

    The CSV preset stores the columns of a model file in their native
    units (velocities in cm/s, angles in degrees).  The returned dict
    always uses the spot API units: velocities in km/s, angles in
    degrees.

    Parameters
    ----------
    name : str
        Preset model name (case-insensitive, e.g. 'hot11', 'FALC11') or
        a path to a CSV file with the same 11 columns.
    ltau : array-like, optional
        Target log10(tau) grid.  If given and different from the preset
        grid, every depth-dependent column is interpolated (with
        extrapolation) onto this grid.
    interpolation : str
        'linear' or 'cubic' (see :mod:`spot.utils.interpolation`).

    Returns
    -------
    dict with keys
        ltau (Nt,), T, Pe, vmic, B, vlos, gamma, phi, Pg, z, rho — all
        arrays of shape (Nt,).
    """
    import numpy as np
    from .interpolation import interp_to_grid

    path = resolve_preset("models", name)
    data = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or row[0].startswith("#") or row[0] == "ltau":
                continue
            for i, key in enumerate(_ATMOS_FIELDS):
                data.setdefault(key, []).append(float(row[i]))
    model = {key: np.asarray(data[key], dtype=float) for key in _ATMOS_FIELDS}

    # model files store velocities in cm/s -> convert to km/s (spot API units)
    for key in ("vmic", "vlos"):
        model[key] = model[key] * 1e-5

    if ltau is not None:
        ltau = np.asarray(ltau, dtype=float)
        if ltau.shape != model["ltau"].shape or not np.allclose(
                ltau, model["ltau"]):
            # model files store ltau DECREASING (index 0 = deepest);
            # the interpolator needs an increasing source grid, so flip
            # both sides and flip the result back to the model-file order.
            src = model["ltau"][::-1]
            if not np.all(np.diff(src) > 0):
                raise ValueError("model ltau must be monotonic")
            tgt = ltau[::-1]
            for key in ("T", "Pe", "vmic", "B", "vlos", "gamma", "phi",
                        "Pg", "z", "rho"):
                model[key] = interp_to_grid(src, model[key][::-1],
                                            tgt, method=interpolation)[::-1]
            model["ltau"] = ltau
    return model


def _as_query(x):
    """Normalise a scalar-or-iterable argument into a list of queries.

    Strings are treated as scalars (not iterated); anything else that
    supports iteration (list/tuple/ndarray/tensor/range) is converted to
    a list.
    """
    if isinstance(x, (str, bytes)):
        return [x]
    if hasattr(x, "__iter__"):
        return list(x)
    return [x]


def search_lines(atom, stage, wavelength, tol=1.0, source="default",
                 return_records=False):
    """
    Search the spectral-line table for lines matching queries given by
    element name, ionisation stage and wavelength.

    ``atom``, ``stage`` and ``wavelength`` may all be scalars OR equal-
    length iterables (e.g. ``(["Fe", "Ca"], [1, 1], [6301.5, 8542.1])``);
    every element ``i`` forms one query ``(atom[i], stage[i],
    wavelength[i])``.  For each query the table entries with the same
    element (case-insensitive) and ionisation stage whose wavelength is
    within ``tol`` [Angstrom] of the requested wavelength are returned;
    results from all queries are merged, deduplicated and sorted.

    Parameters
    ----------
    atom : str or iterable of str
        Element name(s) (e.g. 'Fe', 'Ca').
    stage : int or iterable of int
        Ionisation stage(s) (1 = neutral, 2 = once ionised, ...).
    wavelength : float or iterable of float
        Requested wavelength(s) [Angstrom].
    tol : float, optional
        Wavelength tolerance [Angstrom] (default 1.0; ``None`` disables
        the wavelength filter, returning every line of that element and
        stage).
    source : str, optional
        Preset name (default) or path, forwarded to :func:`load_lines`.
    return_records : bool, optional
        Also return the matching line records.

    Returns
    -------
    list of int or (list of int, list of dict)
        Zero-based indices into ``load_lines(source)`` (sorted,
        deduplicated) usable as ``Synthesis({"lines": indices})``; with
        ``return_records=True`` a second element gives the records.
    """
    atoms = _as_query(atom)
    stages = _as_query(stage)
    wls = _as_query(wavelength)
    if len({len(atoms), len(stages), len(wls)}) > 1:
        raise ValueError(
            "atom/stage/wavelength must be scalars or iterables of the "
            "same length")
    query = zip(atoms, stages, wls)
    all_lines = load_lines(source)
    hits = {}
    for a, s, w in query:
        a = str(a).upper()
        s = int(s)
        w = float(w)
        for i, rec in enumerate(all_lines):
            if rec["atom"].upper() != a or rec["stage"] != s:
                continue
            if tol is not None and abs(rec["wavelength"] - w) > tol:
                continue
            hits[i] = rec
    idxs = sorted(hits)
    if return_records:
        return idxs, [hits[i] for i in idxs]
    return idxs
