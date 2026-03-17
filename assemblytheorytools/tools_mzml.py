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
    """
    Format a string with ANSI color codes and optional bold styling.

    Parameters
    ----------
    msg : str
        The message string to be formatted.
    color : str, optional
        The color name to use for formatting (e.g., "red", "green"). If not in the
        _ANSI_COLORS dictionary or empty, no color is applied.
    bold : bool, optional
        If True, applies bold formatting to the message.

    Returns
    -------
    str
        The formatted string with ANSI color and/or bold codes applied.
    """
    color = _ANSI_COLORS[color] if color in _ANSI_COLORS else ""

    return (
        f'{color}{_ANSI_COLORS["bold"]}{msg}{_ANSI_COLORS["reset"]}'
        if bold
        else f'{color}{msg}{_ANSI_COLORS["reset"]}'
    )


def _make_logger(
        name: str, filename: Optional[str] = "", debug: Optional[bool] = False
) -> logging.Logger:
    """
    Create and configure a logger with optional file and stream handlers.

    This function sets up a logger with a custom ANSI color formatter for both
    console and optional file output. The logger's level is set based on the
    debug flag.

    Parameters
    ----------
    name : str
        The name of the logger.
    filename : str, optional
        If provided, log messages will also be written to this file.
    debug : bool, optional
        If True, sets the logger level to DEBUG; otherwise, INFO.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    # Get logger and set level
    logger = logging.getLogger(name)
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    # Custom ANSI colour formatter
    formatter = _ProtoFormatter()

    # Using file logging, add FileHandler
    if filename:
        fh = logging.FileHandler(filename=filename)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Setup stream handler
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.propagate = False

    return logger


class _ProtoFormatter(logging.Formatter):
    """
    Custom logging formatter that applies ANSI color codes to log messages
    based on their severity level.

    This formatter colors the log level and message differently for DEBUG, INFO,
    WARNING, ERROR, and CRITICAL levels, and includes a timestamp and logger name
    in the output.
    """

    def __init__(self):
        """
        Initialize the _ProtoFormatter by calling the base class constructor.
        """
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the specified log record as a colored string.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to be formatted.

        Returns
        -------
        str
            The formatted log message string with ANSI color codes.
        """
        level, levelno, msg = record.levelname, record.levelno, record.msg
        if levelno == logging.DEBUG:
            level = _colour_item(level, color="red")
        elif levelno == logging.INFO:
            level = _colour_item(level, color="green")
        elif levelno == logging.WARN:
            level = _colour_item(level, color="yellow", bold=True)
            msg = _colour_item(msg, color="yellow")
        elif levelno == logging.ERROR:
            level = _colour_item(level, color="red", bold=True)
            msg = _colour_item(msg, color="red", bold=True)
        elif levelno == logging.CRITICAL:
            level = _colour_item(level, color="red", bold=True)
            msg = _colour_item(msg, color="red")
        timestamp = time.strftime("%d-%m-%Y|%H:%M:%S")
        name = _colour_item(record.name, color="cyan")
        return f"[{timestamp}] - {name}::{level} -- {msg}"


class _UnsupportedCompressionMethod(Exception):
    """
    Exception raised when an unsupported compression method is encountered
    during mzML file parsing.

    This exception should be raised if the code encounters a compression type
    that is not implemented or recognized.

    Examples
    --------
    >>> raise _UnsupportedCompressionMethod("Compression method 'xyz' is not supported.")
    """


class _Spectrum(object):
    """
    Class for representing a Spectrum object from mzML files.

    This class encapsulates the data and methods required to decode, decompress,
    and serialize mass spectrometry spectrum data, including m/z and intensity arrays,
    retention time, precursor information, and more.

    Parameters
    ----------
    intensity_threshold : int
        Threshold for cutting intensities below this value.
    relative : bool, optional
        If True, intensities of individual ions in spectra are displayed as relative (%)
        rather than absolute units. Default is False.
    """

    def __init__(self, intensity_threshold, relative=False):
        """
        Initialize a Spectrum object with default and user-specified parameters.

        Parameters
        ----------
        intensity_threshold : int
            Threshold for cutting intensities below this value.
        relative : bool, optional
            If True, output intensities as relative (%). Default is False.
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
        self.hcd = ""

    def _set_data_type(self):
        """
        Set the data type of the binary data within the spectrum.

        Sets self.d_type to 'f' for 32-bit or 'd' for 64-bit floating point data.
        """
        if "32" in self.d_type:
            self.d_type = "f"
        elif "64" in self.d_type:
            self.d_type = "d"

    def process(self):
        """
        Process the spectrum by decoding and decompressing the m/z and intensity data.

        Decodes the m/z and intensity data from Base64, decompresses if required,
        converts to float arrays, and serializes the spectrum data.
        """
        self._set_data_type()
        self.decode_and_decompress()
        self.serialized = self.serialize()

    def decode_and_decompress(self):
        """
        Decode binary data from Base64 and decompress if necessary.

        Converts the binary m/z and intensity data to lists of floats.
        Raises
        ------
        _UnsupportedCompressionMethod
            If the compression method is not supported.
        """
        # Decode the MZ and intensity data
        self.mz = base64.b64decode(self.mz)
        self.intensity = base64.b64decode(self.intensity)

        # Using ZLib compression
        if "zlib" in self.compression:
            self.mz = self.decompress(self.mz)
            self.intensity = self.decompress(self.intensity)
        else:
            raise _UnsupportedCompressionMethod(
                f"Compression method {self.compression} is not supported."
            )

        # Build the MZ array
        self.mz = list(
            struct.unpack(
                f"<{self.array_length}{self.d_type}",
                self.mz
            )
        )

        # Build the Intensity array
        self.intensity = list(
            struct.unpack(
                f"<{self.array_length}{self.d_type}",
                self.intensity
            )
        )

    def decompress(self, stream: bytes):
        """
        Decompress a data stream using a zlib decompression object.

        Parameters
        ----------
        stream : bytes
            Data stream to decompress.

        Returns
        -------
        bytes
            Decompressed data stream.
        """
        # Decompress the ZLib stream
        zobj = zlib.decompressobj()
        stream = zobj.decompress(stream)
        return stream + zobj.flush()

    def serialize(self) -> Dict:
        """
        Convert the spectrum into a dictionary containing relevant information.

        Only includes peaks above the intensity threshold and relevant metadata.

        Returns
        -------
        dict
            Spectrum data, including m/z, intensity, retention time, parent info, etc.
        """
        out = {}
        mass_list = []

        # Iterate through the MZ and intensity
        for mz, intensity in zip(self.mz, self.intensity):
            # Check the intensity threshold is met and add to output
            if self.ms_level == "1":
                if intensity > self.intensity_threshold:
                    out[f"{mz:.4f}"] = int(intensity)
                    mass_list.append(mz)

            # Check the intensity threshold is met and add to output for MS 2+
            elif self.ms_level > "1":
                if intensity > (self.intensity_threshold / 100) * 5:
                    out[f"{mz:.4f}"] = int(intensity)
                    mass_list.append(mz)

        # Populate remaining data
        out["retention_time"] = self.retention_time
        out["scan"] = self.scan
        out['hcd'] = self.hcd

        # Set the parent mass if applicable
        if self.parent_mass:
            # AMK changed this:
            # out["parent"] = f"{float(self.parent_mass):.4f}"
            out["parent"] = f"{float(self.precursors[0]):.4f}"

        # AMK: Set the precursor list
        if self.precursors:
            out["precursors"] = self.precursors

        # Set parent scan if applicable
        if self.parent_scan:
            # AMK changed this:
            # out["parent_scan"] = self.parent_scan
            out["parent_scan"] = self.precursors_scans[0]

        # AMK: Set the precursor list
        if self.precursors_scans:
            out["precursors_scans"] = self.precursors_scans

        # AMK: Set fragmentation energy
        if self.hcd:
            out["HCD"] = self.hcd

        # Create mass list
        out["mass_list"] = [float(f"{mass:.4f}") for mass in mass_list]

        # If relative intensities are to be returned, convert spectrum dict
        if self.relative:
            out = self.convert_to_relative(out)

        return out

    def convert_to_relative(self, spectrum_dict: dict) -> Dict:
        """
        Convert a spectrum dictionary of absolute intensities to relative intensities.

        Parameters
        ----------
        spectrum_dict : dict
            Standard spectrum dictionary with absolute intensities.

        Returns
        -------
        dict
            Spectrum data with relative intensities and base peak information.
        """
        #  get list of ions ([[m/z, I], ...]) sorted by intensity
        all_ions = sorted([
            [float(mass), float(intensity)]
            for mass, intensity in spectrum_dict.items()
            if mass not in _NON_MASS_KEYS], key=lambda x: x[1])

        #  get the base peak - most intense ion
        base_peak = all_ions[-1]

        #  make sure all _NON_MASS_KEYS remain unchanged in spectrum_dict
        spectrum_dict = {
            key: value for key, value in spectrum_dict.items()
            if key in _NON_MASS_KEYS
        }

        #  iterate through ions, readding to spectrum_dict with relative intensities
        for ion in all_ions:
            spectrum_dict[ion[0]] = round((ion[1] / base_peak[1]) * 100, 4)
        spectrum_dict["base_peak"] = base_peak

        return spectrum_dict


def _create_regex_mapper() -> dict:
    """
    Create a mapping of XML tag names to their corresponding regular expression patterns.

    This utility function returns a dictionary where each key is a descriptive string
    for a particular mzML XML attribute or element, and each value is a regular expression
    string that can be used to extract the corresponding value from a line of mzML text.

    Returns
    -------
    dict
        Mapping of tag names to regular expression patterns for extracting values from mzML lines.
        Keys include:
            - "spec_index": Regex for spectrum index attribute.
            - "array_length": Regex for default array length attribute.
            - "value": Regex for value attribute.
            - "name": Regex for name attribute.
            - "binary": Regex for binary data between <binary> tags.
            - "scan": Regex for scan number attribute.
    """
    return {
        "spec_index": r'index="(.+?)"',
        "array_length": r'defaultArrayLength="(.+?)"',
        "value": r'value="(.+?)"',
        "name": r'name="(.+?)"',
        "binary": r"<binary>(.*?)</binary>",
        "scan": r"scan=([0-9]+)",
    }


def _value_finder(regex: str, line: str) -> str:
    """
    Search for a value in a string using a regular expression.

    This function applies the provided regular expression to the input line and returns
    the first captured group if a match is found. If no match is found, it returns None.

    Parameters
    ----------
    regex : str
        The regular expression pattern to search for.
    line : str
        The string to search within.

    Returns
    -------
    str or None
        The matched value (first capture group) if found, otherwise None.
    """
    result = re.search(regex, line)

    if result:
        return result.group(1)
    return None


def _write_json(data: dict, filename: str):
    """
    Write a dictionary to a JSON file.

    This function serializes the provided dictionary and writes it to the specified
    file in JSON format with indentation for readability.

    Parameters
    ----------
    data : dict
        The data to write to the JSON file.
    filename : str
        The name (or path) of the file to write the JSON data to.

    Returns
    -------
    None
    """
    with open(filename, "w") as f_d:
        json.dump(data, f_d, indent=4)


def _banned_phrases(line: str) -> bool:
    """
    Check if any banned phrase exists in the given line.

    This function iterates through a list of banned phrases and checks if any of them
    are present in the input line. Banned phrases are those that can interfere with
    parsing and indicate that the parser should ignore the line.

    Parameters
    ----------
    line : str
        The line of text to check for banned phrases.

    Returns
    -------
    bool
        True if a banned phrase exists in the line, False otherwise.
    """
    # Iterate through all banned phrases
    for phrase in _BANNED_PHRASES:
        # Phrase is banned
        if phrase in line:
            return True

    # No banned phrases found
    return False


class _InvalidInputFile(Exception):
    """
    Exception raised for invalid input file formats.

    This exception should be raised when a file provided to the mzML parser
    does not exist, is not a file, or does not have the expected '.mzML' extension.
    """


class _MzmlParser:
    """
    Class for parsing an mzML file and extracting MS spectra data.

    This parser reads an mzML file, extracts all MS1 and MS2 spectra along with
    retention time, parent mass, and other relevant metadata, and can output the
    results as a JSON file. It supports multi-threaded processing of spectra and
    handles both absolute and relative intensity representations.

    Parameters
    ----------
    filename : str
        Name of the mzML file to parse.
    output_dir : str
        Directory where the output JSON file will be saved.
    rt_units : int or None, optional
        Retention time units. Defaults to None.
    int_threshold : int, optional
        Intensity threshold for filtering peaks. Defaults to 1000.
    relative_intensity : bool, optional
        If True, output intensities as relative (%). If False, use absolute intensities.
        Defaults to False.

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
    rt_units : int or None
        Retention time units.
    """

    def __init__(
            self,
            filename: str,
            output_dir: str,
            rt_units: Optional[int] = None,
            int_threshold: Optional[int] = 1000,
            relative_intensity: Optional[bool] = False,
    ):
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

    def _check_file(self):
        """
        Check if the input file is valid for parsing.

        Ensures the file exists, is a file, and has the '.mzML' extension.

        Raises
        ------
        _InvalidInputFile
            If the file does not exist, is not a file, or does not have the correct extension.
        """
        if not os.path.isfile(self.filename) or not self.filename.endswith(
                ".mzML"
        ):
            raise _InvalidInputFile(f"File {self.filename} is not valid!")

    def parse_file(self) -> Dict:
        """
        Parse the mzML file and extract all spectra information.

        Reads the file line by line, processes each spectrum, and organizes the
        data by MS level. The spectra are then processed and written to a JSON file.

        Returns
        -------
        dict
            Dictionary of each spectrum split by MS level.
        """
        # Check the file exists and is an mzML file
        self._check_file()

        # Open the file and process each line individually
        with open(self.filename) as f_d:
            self.logger.info(
                f"Parsing file: {_colour_item(self.filename, 'yellow')}..."
            )
            for line in f_d.readlines():
                self.process_line(line)

        self.logger.info(
            f"Parsing complete!\nTotal Spectra:\
            {_colour_item(str(len(self.spectra)), 'green')}"
        )
        self.logger.info("Processing spectra...")

        # Get all MS level spectra from the collection
        ms_levels = [
            [spec for spec in self.spectra if spec.ms_level == str(level)]
            for level in range(1, max(map(int, list(self.ms.keys()))) + 1)
        ]

        # Process and write out to file
        self.bulk_process(*ms_levels)
        output = self.write_out_to_file()
        self.logger.info(f"{_colour_item('Complete', 'green')}")

        return output

    def bulk_process(self, *ms_levels: List[_Spectrum]):
        """
        Create threads for processing MS1 and MS2 data simultaneously.

        Parameters
        ----------
        ms_levels : list of list of _Spectrum
            Collections of MS spectra, one list per MS level.
        """
        pool = [
            Thread(target=self.process_spectra, args=(ms,)) for ms in ms_levels
        ]

        [thread.start() for thread in pool]
        [thread.join() for thread in pool]

    def process_spectra(self, spectra: List[_Spectrum]):
        """
        Process spectra from a list and serialize the data.

        Parameters
        ----------
        spectra : list of _Spectrum
            List of Spectrum objects to process and serialize.
        """
        for spec in spectra:
            spec.process()
            self.ms[spec.ms_level].append(spec)

    def build_output(self) -> Dict:
        """
        Build the MS data output from the processed spectra.

        Sorts spectra by retention time and organizes them by MS level.

        Returns
        -------
        dict
            MS spectra split by level, with each spectrum serialized.
        """
        # Create the output
        output = {"ms" + str(x): {} for x in self.ms.keys()}

        # Sort the MS spectra by retention time
        for ms_level in self.ms:
            self.ms[ms_level] = sorted(
                self.ms[ms_level], key=lambda x: x.retention_time
            )

        # Populate the output
        for ms_level in sorted(list(self.ms.keys())):
            for pos, spec in enumerate(self.ms[ms_level]):
                if not spec.serialized:
                    spec.rma_process()
                if spec.serialized["mass_list"]:
                    output["ms" + ms_level][
                        f"spectrum_{pos + 1}"
                    ] = spec.serialized

        return output

    def write_out_to_file(self):
        """
        Write the processed MS1 and MS2 data to a JSON file.

        If any spectra are not processed, they are processed here before writing.

        Returns
        -------
        dict
            The output dictionary that was written to file.
        """
        output = self.build_output()

        name = self.filename.split(os.sep)[-1]

        name = "ripper_" + name
        out_path = os.path.join(
            self.output_dir, name.replace(".mzML", ".json")
        )

        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        _write_json(output, out_path)

        return output

    def process_line(self, line: str):
        """
        Process a single line from the mzML file.

        Determines whether the line starts a new spectrum, ends a spectrum,
        or contains relevant information to extract.

        Parameters
        ----------
        line : str
            Line from the mzML file.
        """
        # Currently not in a spectrum, set the spectrum flag
        if not self.in_spectrum:
            self.start_spectrum(line)

        # Look for end of spectrum tag
        else:
            if "</spectrum>" in line:
                self.spectra.append(self.spec)
                self.in_spectrum = False
                self.spec = _Spectrum(
                    intensity_threshold=self.spec_int_threshold,
                    relative=self.relative,
                )
            else:
                if _banned_phrases(line):
                    return

                self.extract_information(line)

    def start_spectrum(self, line: str):
        """
        Initiate the spectrum data gathering process.

        Checks for a spectrum index tag and, if found, initializes a new spectrum.

        Parameters
        ----------
        line : str
            Line from the mzML file.
        """
        # Extract the spectrum ID
        spec_id = _value_finder(self.re_expr["spec_index"], line)
        if not spec_id:
            return

        # Set the flag and ID
        self.in_spectrum = True
        self.spec.id = spec_id

        # Find the size of the data array
        self.spec.array_length = _value_finder(self.re_expr["array_length"], line)

    def extract_information(self, line: str):
        """
        Attempt to extract information from a given line.

        Extracts retention time, data type, compression, m/z, intensity, and
        other relevant spectrum information.

        Parameters
        ----------
        line : str
            Line from the mzML file.

        Raises
        ------
        Exception
            If unable to determine what kind of binary data is being processed.
        """
        # MS Level
        if "MS:1000511" in line:
            self.spec.ms_level = _value_finder(self.re_expr["value"], line)
            if self.spec.ms_level not in self.ms:
                self.ms[self.spec.ms_level] = []

        # Scan Number
        elif "MS:1000796" in line:
            self.spec.scan = _value_finder(self.re_expr["scan"], line)

        # Retention time
        elif "MS:1000016" in line:
            rt_converter = 1
            if self.rt_units == "sec":
                rt_converter = 60
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
            self.spec.precursors.append(
                _value_finder(self.re_expr["value"], line)
            )

        # Parent Scan
        elif "<precursor spectrumRef" in line:
            self.spec.parent_scan = _value_finder(self.re_expr["scan"], line)
            self.spec.precursors_scans.append(
                _value_finder(self.re_expr["scan"], line)
            )

        # Suggested parent mass
        elif "MS:1000512" in line:
            suggested_parent = _value_finder(self.re_expr["value"], line)
            self.update_parent(suggested_parent)

        # MZ data
        elif "MS:1000514" in line:
            self.curr_spec_bin_type = 0

        # Intensity data
        elif "MS:1000515" in line:
            self.curr_spec_bin_type = 1

        # Binary blob
        elif "<binary>" in line:
            binary_text = _value_finder(self.re_expr["binary"], line)

            # Looking at MZ values
            if self.curr_spec_bin_type == 0:
                self.spec.mz = binary_text

            # Looking at intensity values
            elif self.curr_spec_bin_type == 1:
                self.spec.intensity = binary_text

            # No idea what we're looking at
            else:
                raise Exception("Error setting binary type")

        # Nothing
        else:
            return

    def update_parent(self, filter_string: str):
        """
        Update the parent mass for MS3 and above spectra.

        Parameters
        ----------
        filter_string : str
            String containing parent information, typically from the mzML filter line.
        """
        # Below MS level 3
        if int(self.spec.ms_level) < 3:
            return

        # Sets the parent for MS levels 3 and above
        parents = filter_string.split("@")
        self.spec.parent_mass = parents[int(self.spec.ms_level) - 2].split(
            " "
        )[-1]
        # if self.spec.ms_level == "3":
        #     self.spec.parent_mass = parents[1].split(" ")[-1]
        # elif self.spec.ms_level == "4":
        #     self.spec.parent_mass = parents[2].split(" ")[-1]


def process_mzml_file(
        filename: str,
        out_dir: str,
        rt_units=None,
        int_threshold=1000,
        relative=False,
):
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
