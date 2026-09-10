"""
Parsing of mzML mass spectrometry files.

This module decodes the binary data arrays embedded in mzML documents, handling
the supported compression methods and precisions, and exposes the resulting
spectra for downstream assembly analysis.
"""

import base64
import json
import logging
import os
import re
import struct
import time
import zlib
from threading import Thread
from typing import Dict, List, Optional

_ANSI_COLORS = {
    "black": "\u001b[30m",
    "red": "\u001b[31m",
    "green": "\u001b[32m",
    "yellow": "\u001b[33m",
    "blue": "\u001b[34m",
    "magenta": "\u001b[35m",
    "cyan": "\u001b[36m",
    "white": "\u001b[37m",
    "bold": "\u001b[1m",
    "reset": "\u001b[0m",
}

_NON_MASS_KEYS = ["mass_list", "retention_time", "parent", "scan", "parent_scan", "hcd"]

_BANNED_PHRASES = ["<userParam"]


def _colour_item(
    msg: str, color: Optional[str] = "", bold: Optional[bool] = False
) -> str:
    """Wrap a message in ANSI color and optional bold codes, then reset styling."""
    color = _ANSI_COLORS.get(color, "")
    weight = _ANSI_COLORS["bold"] if bold else ""
    return f'{color}{weight}{msg}{_ANSI_COLORS["reset"]}'


def _make_logger(
    name: str, filename: Optional[str] = "", debug: Optional[bool] = False
) -> logging.Logger:
    """Configure a non-propagating logger with a stream handler and optional file."""
    logger = logging.getLogger(name)
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)
    formatter = _ProtoFormatter()

    handlers = [logging.FileHandler(filename=filename)] if filename else []
    handlers.append(logging.StreamHandler())
    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


class _ProtoFormatter(logging.Formatter):
    """Format log records with a timestamp, logger name, and severity colors."""

    def format(self, record: logging.LogRecord) -> str:
        """Apply severity colors to the original record message and level name."""
        level, levelno, msg = record.levelname, record.levelno, record.msg
        if levelno == logging.DEBUG:
            level = _colour_item(level, color="red")
        elif levelno == logging.INFO:
            level = _colour_item(level, color="green")
        elif levelno == logging.WARN:
            level = _colour_item(level, color="yellow", bold=True)
            msg = _colour_item(msg, color="yellow")
        elif levelno in (logging.ERROR, logging.CRITICAL):
            level = _colour_item(level, color="red", bold=True)
            msg = _colour_item(msg, color="red", bold=levelno == logging.ERROR)
        timestamp = time.strftime("%d-%m-%Y|%H:%M:%S")
        name = _colour_item(record.name, color="cyan")
        return f"[{timestamp}] - {name}::{level} -- {msg}"


class _UnsupportedCompressionMethod(Exception):
    """Raised when binary arrays declare an unsupported compression method."""


class _Spectrum:
    """
    A single spectrum read from an mzML file.

    This class encapsulates the data and methods required to decode,
    decompress, and serialize mass spectrometry spectrum data, including m/z
    and intensity arrays, retention time, precursor information, and more.

    Attributes
    ----------
    scan : str
        Scan number of the spectrum.
    array_length : str
        Number of points in the binary m/z and intensity arrays.
    ms_level : str
        MS level of the spectrum ("1" for MS1, "2" for MS2, and so on).
    precursors : list
        Precursor m/z values recorded for the spectrum.
    precursors_scans : list
        Scan numbers of the precursor spectra.
    parent_mass : str
        m/z of the parent ion the spectrum was fragmented from.
    parent_scan : str
        Scan number of the parent spectrum.
    retention_time : str
        Retention time of the spectrum, in minutes.
    d_type : str
        Binary data type, normalised to ``'f'`` (32-bit) or ``'d'`` (64-bit)
        by :meth:`_set_data_type`.
    compression : str
        Compression method declared for the binary arrays.
    mz : str or list of float
        Base64 m/z array, replaced by the decoded values once
        :meth:`decode_and_decompress` has run.
    intensity : str or list of float
        Base64 intensity array, replaced by the decoded values once
        :meth:`decode_and_decompress` has run.
    hcd : str
        Collision energy used to acquire the spectrum.
    serialized : dict
        The m/z to intensity mapping produced by :meth:`serialize`.
    intensity_threshold : int
        Threshold below which intensities are discarded.
    relative : bool
        Whether intensities are reported relative to the base peak.
    """

    def __init__(self, intensity_threshold: int, relative: bool = False) -> None:
        """
        Create an empty spectrum container.

        All spectrum fields start empty and are filled in as the mzML file
        is parsed; only the intensity handling options are set here.

        Parameters
        ----------
        intensity_threshold : int
            Threshold for cutting intensities below this value.
        relative : bool, optional
            If True, intensities of individual ions in spectra are displayed
            as relative (%) rather than absolute units. Default is False.
        """
        self.scan = ""
        self.array_length = ""
        self.ms_level = ""
        self.precursors = []
        self.precursors_scans = []
        self.parent_mass = ""
        self.parent_scan = ""
        self.retention_time = ""
        self.d_type = ""
        self.compression = ""
        self.mz = ""
        self.intensity = ""
        self.hcd = ""
        self.serialized = {}
        self.intensity_threshold = intensity_threshold
        self.relative = relative

    def _set_data_type(self) -> None:
        """Normalize 32-bit and 64-bit float descriptions to struct format codes."""
        if "32" in self.d_type:
            self.d_type = "f"
        elif "64" in self.d_type:
            self.d_type = "d"

    def process(self) -> None:
        """Decode both binary arrays and store the serialized spectrum."""
        self._set_data_type()
        self.decode_and_decompress()
        self.serialized = self.serialize()

    def decode_and_decompress(self) -> None:
        """Decode base64/zlib arrays as little-endian floats.

        Raises _UnsupportedCompressionMethod for compression other than zlib.
        """
        self.mz = base64.b64decode(self.mz)
        self.intensity = base64.b64decode(self.intensity)

        if "zlib" not in self.compression:
            raise _UnsupportedCompressionMethod(
                f"Compression method {self.compression} is not supported."
            )

        self.mz = self.decompress(self.mz)
        self.intensity = self.decompress(self.intensity)
        array_format = f"<{self.array_length}{self.d_type}"
        self.mz = list(struct.unpack(array_format, self.mz))
        self.intensity = list(struct.unpack(array_format, self.intensity))

    def decompress(self, stream: bytes) -> bytes:
        """Decompress a zlib stream, including any buffered output."""
        decompressor = zlib.decompressobj()
        return decompressor.decompress(stream) + decompressor.flush()

    def serialize(self) -> Dict:
        """Return filtered peaks and metadata, optionally using relative intensities.

        MS1 peaks must exceed the intensity threshold; higher MS levels use
        five percent of that threshold. Mass keys and mass_list entries are
        rounded to four decimal places, and absolute intensities are integers.
        """
        out = {}
        mass_list = []

        for mz, intensity in zip(self.mz, self.intensity):
            if self.ms_level == "1":
                threshold = self.intensity_threshold
            elif self.ms_level > "1":
                threshold = (self.intensity_threshold / 100) * 5
            else:
                continue
            if intensity > threshold:
                out[f"{mz:.4f}"] = int(intensity)
                mass_list.append(mz)

        out["retention_time"] = self.retention_time
        out["scan"] = self.scan
        out["hcd"] = self.hcd

        if self.parent_mass:
            out["parent"] = f"{float(self.precursors[0]):.4f}"
        if self.precursors:
            out["precursors"] = self.precursors
        if self.parent_scan:
            out["parent_scan"] = self.precursors_scans[0]
        if self.precursors_scans:
            out["precursors_scans"] = self.precursors_scans
        if self.hcd:
            out["HCD"] = self.hcd

        out["mass_list"] = [float(f"{mass:.4f}") for mass in mass_list]
        return self.convert_to_relative(out) if self.relative else out

    def convert_to_relative(self, spectrum_dict: dict) -> Dict:
        """Return peaks in ascending intensity order, as percentages of the base peak.

        Preserve metadata listed in _NON_MASS_KEYS and include the original
        [m/z, intensity] pair under base_peak.
        """
        all_ions = sorted(
            (
                [float(mass), float(intensity)]
                for mass, intensity in spectrum_dict.items()
                if mass not in _NON_MASS_KEYS
            ),
            key=lambda ion: ion[1],
        )
        base_peak = all_ions[-1]

        spectrum_dict = {
            key: value
            for key, value in spectrum_dict.items()
            if key in _NON_MASS_KEYS
        }
        for mass, intensity in all_ions:
            spectrum_dict[mass] = round((intensity / base_peak[1]) * 100, 4)
        spectrum_dict["base_peak"] = base_peak

        return spectrum_dict


def _create_regex_mapper() -> dict:
    """Return the XML attribute and binary-data patterns used by the parser."""
    return {
        "spec_index": r'index="(.+?)"',
        "array_length": r'defaultArrayLength="(.+?)"',
        "value": r'value="(.+?)"',
        "name": r'name="(.+?)"',
        "binary": r"<binary>(.*?)</binary>",
        "scan": r"scan=([0-9]+)",
    }


def _value_finder(regex: str, line: str) -> str:
    """Return the first captured value, or None when the pattern does not match."""
    result = re.search(regex, line)
    return result.group(1) if result else None


def _write_json(data: dict, filename: str) -> None:
    """Write data as indented JSON to filename."""
    with open(filename, "w") as output_file:
        json.dump(data, output_file, indent=4)


def _banned_phrases(line: str) -> bool:
    """Return whether the line contains a phrase the parser should ignore."""
    return any(phrase in line for phrase in _BANNED_PHRASES)


class _InvalidInputFile(Exception):
    """Raised when the input is not an existing file with the .mzML extension."""


class _MzmlParser:
    """
    Parser for an mzML file, extracting MS spectra data.

    This parser reads an mzML file, extracts all MS1 and MS2 spectra along
    with retention time, parent mass, and other relevant metadata, and can
    output the results as a JSON file. It supports multi-threaded processing
    of spectra and handles both absolute and relative intensity
    representations.

    Attributes
    ----------
    logger : logging.Logger
        Logger for reporting progress and errors.
    filename : str
        Path to the mzML file.
    output_dir : str
        Output directory for JSON results.
    in_spectrum : bool
        Flag indicating if currently parsing a spectrum.
    re_expr : dict
        Dictionary of regex patterns for parsing mzML lines.
    spectra : list
        List of parsed _Spectrum objects.
    ms : dict
        Dictionary mapping MS levels to lists of _Spectrum objects.
    spec : _Spectrum
        The current spectrum being parsed.
    relative : bool
        Whether to output relative intensities.
    spec_int_threshold : int
        Intensity threshold for filtering peaks.
    curr_spec_bin_type : int
        Indicator for current binary data type (m/z or intensity).
    rt_units : str or None
        Retention time units.
    """

    def __init__(
        self,
        filename: str,
        output_dir: str,
        rt_units: Optional[str] = None,
        int_threshold: Optional[int] = 1000,
        relative_intensity: Optional[bool] = False,
    ) -> None:
        """
        Create a parser for a single mzML file.

        Sets up the logger, the regex mapper used to read mzML lines, and
        the empty spectrum containers the parse fills in.

        Parameters
        ----------
        filename : str
            Path to the mzML file to parse.
        output_dir : str
            Directory the extracted spectra are written to. The path is
            resolved to an absolute path.
        rt_units : str or None, optional
            Retention time units used by the source file. Only ``"sec"``
            triggers a conversion (divides by 60); any other value,
            including the default ``None``, is treated as already being in
            minutes.
        int_threshold : int, optional
            Minimum intensity for a peak to be retained. Defaults to 1000.
        relative_intensity : bool, optional
            If True, report intensities relative to the base peak rather
            than as absolute values. Defaults to False.
        """
        self.logger = _make_logger("MzMLRipper")
        self.filename = filename
        self.output_dir = os.path.abspath(output_dir)
        self.in_spectrum = False
        self.re_expr = _create_regex_mapper()
        self.spectra = []
        self.ms = {}

        self.spec = _Spectrum(
            intensity_threshold=int_threshold, relative=relative_intensity
        )
        self.relative = relative_intensity
        self.spec_int_threshold = int_threshold
        self.curr_spec_bin_type = -1
        self.rt_units = rt_units

    def _check_file(self) -> None:
        """Raise _InvalidInputFile unless the input is an existing .mzML file."""
        if not os.path.isfile(self.filename) or not self.filename.endswith(".mzML"):
            raise _InvalidInputFile(f"File {self.filename} is not valid!")

    def parse_file(self) -> Dict:
        """Read spectra, process them by MS level, and return the saved JSON data."""
        self._check_file()

        with open(self.filename) as input_file:
            self.logger.info(
                f"Parsing file: {_colour_item(self.filename, 'yellow')}..."
            )
            for line in input_file:
                self.process_line(line)

        self.logger.info(
            "Parsing complete!\nTotal Spectra:            "
            f"{_colour_item(str(len(self.spectra)), 'green')}"
        )
        self.logger.info("Processing spectra...")

        ms_levels = [
            [spec for spec in self.spectra if spec.ms_level == str(level)]
            for level in range(1, max(map(int, self.ms)) + 1)
        ]

        self.bulk_process(*ms_levels)
        output = self.write_out_to_file()
        self.logger.info(_colour_item("Complete", "green"))

        return output

    def bulk_process(self, *ms_levels: List[_Spectrum]) -> None:
        """Process each collection of spectra in a separate thread and wait for all."""
        pool = [Thread(target=self.process_spectra, args=(ms,)) for ms in ms_levels]

        for thread in pool:
            thread.start()
        for thread in pool:
            thread.join()

    def process_spectra(self, spectra: List[_Spectrum]) -> None:
        """Decode and serialize spectra, appending each to its MS-level collection."""
        for spec in spectra:
            spec.process()
            self.ms[spec.ms_level].append(spec)

    def build_output(self) -> Dict:
        """Return serialized spectra grouped by MS level and sorted by retention time.

        Retention times retain their stored ordering (normally strings).
        Empty spectra are omitted without renumbering the remaining spectra.
        """
        output = {f"ms{level}": {} for level in self.ms}

        for ms_level, spectra in self.ms.items():
            self.ms[ms_level] = sorted(
                spectra, key=lambda spec: spec.retention_time
            )

        for ms_level in sorted(self.ms):
            for position, spec in enumerate(self.ms[ms_level], start=1):
                if not spec.serialized:
                    spec.process()
                if spec.serialized["mass_list"]:
                    output["ms" + ms_level][f"spectrum_{position}"] = spec.serialized

        return output

    def write_out_to_file(self) -> Dict:
        """Save and return the output as ripper_<input stem>.json in output_dir."""
        output = self.build_output()

        name = "ripper_" + os.path.basename(self.filename)
        out_path = os.path.join(self.output_dir, name.replace(".mzML", ".json"))

        output_dir = os.path.dirname(out_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        _write_json(output, out_path)

        return output

    def process_line(self, line: str) -> None:
        """Start or finish a spectrum, or extract information from an allowed line."""
        if not self.in_spectrum:
            self.start_spectrum(line)
        elif "</spectrum>" in line:
            self.spectra.append(self.spec)
            self.in_spectrum = False
            self.spec = _Spectrum(
                intensity_threshold=self.spec_int_threshold,
                relative=self.relative,
            )
        elif not _banned_phrases(line):
            self.extract_information(line)

    def start_spectrum(self, line: str) -> None:
        """Begin a spectrum when its index is found and record its array length."""
        spec_id = _value_finder(self.re_expr["spec_index"], line)
        if not spec_id:
            return

        self.in_spectrum = True
        self.spec.id = spec_id
        self.spec.array_length = _value_finder(self.re_expr["array_length"], line)

    def extract_information(self, line: str) -> None:
        """Extract the first recognized spectrum field from a line.

        Raise Exception if binary data appears before its array type is known.
        """
        # MS Level
        if "MS:1000511" in line:
            self.spec.ms_level = _value_finder(self.re_expr["value"], line)
            self.ms.setdefault(self.spec.ms_level, [])

        # Scan Number
        elif "MS:1000796" in line:
            self.spec.scan = _value_finder(self.re_expr["scan"], line)

        # Retention time
        elif "MS:1000016" in line:
            rt_converter = 60 if self.rt_units == "sec" else 1
            self.spec.retention_time = str(
                float(_value_finder(self.re_expr["value"], line)) / rt_converter
            )

        # Fragmentation energy
        elif "MS:1000512" in line:
            self.spec.hcd = (
                _value_finder(self.re_expr["value"], line)
                .split("hcd")[-1]
                .split(" ")[0]
            )

        # Data type (32 or 64 bit)
        elif "MS:1000521" in line or "MS:1000523" in line:
            self.spec.d_type = _value_finder(self.re_expr["name"], line)

        # Compression type
        elif "MS:1000574" in line:
            self.spec.compression = _value_finder(self.re_expr["name"], line)

        # Parent mass
        elif "MS:1000744" in line:
            self.spec.parent_mass = _value_finder(self.re_expr["value"], line)
            self.spec.precursors.append(self.spec.parent_mass)

        # Parent Scan
        elif "<precursor spectrumRef" in line:
            self.spec.parent_scan = _value_finder(self.re_expr["scan"], line)
            self.spec.precursors_scans.append(self.spec.parent_scan)

        # MZ data
        elif "MS:1000514" in line:
            self.curr_spec_bin_type = 0

        # Intensity data
        elif "MS:1000515" in line:
            self.curr_spec_bin_type = 1

        # Binary blob
        elif "<binary>" in line:
            binary_text = _value_finder(self.re_expr["binary"], line)

            if self.curr_spec_bin_type == 0:
                self.spec.mz = binary_text
            elif self.curr_spec_bin_type == 1:
                self.spec.intensity = binary_text
            else:
                raise Exception("Error setting binary type")

    def update_parent(self, filter_string: str) -> None:
        """Read the parent mass from a filter string for MS3 and higher spectra."""
        ms_level = int(self.spec.ms_level)
        if ms_level < 3:
            return

        parents = filter_string.split("@")
        self.spec.parent_mass = parents[ms_level - 2].split(" ")[-1]


def process_mzml_file(
    filename: str,
    out_dir: str,
    rt_units: Optional[str] = "min",
    int_threshold: int = 1000,
    relative: bool = False,
) -> Dict:
    """
    Process an mzML file and extract MS spectra data, saving the results as JSON.

    This function initializes an _MzmlParser instance with the provided parameters,
    parses the mzML file, and writes the extracted MS1 and MS2 spectra to a JSON file
    in the specified output directory.

    Parameters
    ----------
    filename : str
        Path to the mzML file to be processed.
    out_dir : str
        Directory where the output JSON file will be saved.
    rt_units : str or None, optional
        Retention time units. If None, the default units are used.
    int_threshold : int, optional
        Intensity threshold for filtering peaks. Defaults to 1000.
    relative : bool, optional
        If True, output intensities as relative (%). If False, use absolute intensities.

    Returns
    -------
    dict
        Dictionary containing the processed MS spectra data, split by MS level.
    """
    return _MzmlParser(
        filename,
        out_dir,
        rt_units=rt_units,
        int_threshold=int_threshold,
        relative_intensity=relative,
    ).parse_file()
