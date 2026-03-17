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
    color = _ANSI_COLORS[color] if color in _ANSI_COLORS else ""

    return (
        f'{color}{_ANSI_COLORS["bold"]}{msg}{_ANSI_COLORS["reset"]}'
        if bold
        else f'{color}{msg}{_ANSI_COLORS["reset"]}'
    )


def _make_logger(
        name: str, filename: Optional[str] = "", debug: Optional[bool] = False
) -> logging.Logger:
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
    def __init__(self):
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
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
    """Compression type is not yet supported
    """


class _Spectrum(object):
    """Class for representing a Spectrum object from mzML

    Arguments:
        intensity_threshold (int): Threshold for cutting intensities below
        threshold
        relative (bool, optional): Specifies whether intensities of individual
            ions in spectra are displayed in relative (%) or absolute units.
    """

    def __init__(self, intensity_threshold, relative=False):
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
        """Sets the data type of the binary data within
        """

        if "32" in self.d_type:
            self.d_type = "f"
        elif "64" in self.d_type:
            self.d_type = "d"

    def process(self):
        """Processes a Spectrum

        Decodes the m/z and intensity data from Base64
        Decompresses if required and converts to float array
        """

        self._set_data_type()
        self.decode_and_decompress()
        self.serialized = self.serialize()

    def decode_and_decompress(self):
        """Decodes binary data from Base64 and decompresses if necessary

        Converts the binary data to a list of floats
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
        Decompresses a data stream using a zlib decompression object.
        Args:
            stream (bytes): data stream.

        Returns:
            bytes: decompressed data stream.
        """

        # Decompress the ZLib stream
        zobj = zlib.decompressobj()
        stream = zobj.decompress(stream)
        return stream + zobj.flush()

    def serialize(self) -> Dict:
        """Converts the spectrum into a dictionary

        Only takes relevant information

        Returns:
            Dict: Spectrum data
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

            # Check the intensity threshold is met and add to output for MS 2
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

        #  if relative intensities are to be returned, convert spectrum dict
        if self.relative:
            out = self.convert_to_relative(out)

        return out

    def convert_to_relative(self, spectrum_dict: dict) -> Dict:
        """Converts a spectrum dict of absolute intensities to relative
        intensities.

        spectrum_dict (dict): standard spectrum dict with absolute intensities.
        Returns:
            Dict: Spectrum data
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

        #  iterate through ions, readding to spectrum_dict with relative
        #  intensities
        for ion in all_ions:
            spectrum_dict[ion[0]] = round((ion[1] / base_peak[1]) * 100, 4)
        spectrum_dict["base_peak"] = base_peak

        return spectrum_dict


def _create_regex_mapper() -> dict:
    """Creates a mapping of tags to RegEx strings

    Returns:
        dict -- Mapping of tags to RegEx
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
    """Finds a value using RegEx from a given line

    Returns None if nothing found

    Arguments:
        regex {str} -- RegEx string
        line {str} -- Line to parse

    Returns:
        str -- Match if found, None if not
    """

    result = re.search(regex, line)

    if result:
        return result.group(1)
    return None


def _write_json(data: dict, filename: str):
    """Writes data to JSON file

    Arguments:
        data {dict} -- Data to write
        filename {str} -- Name fo the file
    """

    with open(filename, "w") as f_d:
        json.dump(data, f_d, indent=4)


def _banned_phrases(line: str) -> bool:
    """Small check to determine if any banned phrase is in a given line.
    Banned phrases are phrases that can interfere with the parsing and indicate
    that the parser should ignore said line.

    Args:
        line (str): Line to check

    Returns:
        bool: Banned phrase exists
    """

    # Iterate through all banned phrases
    for phrase in _BANNED_PHRASES:
        # Phrase is banned
        if phrase in line:
            return True

    # No banned phrases found
    return False


class _InvalidInputFile(Exception):
    """Exception for invalid file formats"""


class _MzmlParser:
    """Class for parsing an mzML file.

    Extracts all MS1 and MS2 data, along with retention time and parent mass

    Args:
        filename (str): Name of the file to parse
        output_dir (str): Location of where to save the JSON file
        rt_units (int, optional): Retention time units. Defaults to `None`
        int_threshold (int, optional): Intensity Threshold. Defaults to 1000.
        relative_intensity (bool, optional): Specifies whether final
            intensities for individual ions in spectra are displayed as
            relative (%) or absolute intensities. Defaults to False.
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
        """Checks if a file is valid for the parser
        Checks if the file is actually a file and if it is an mzML file

        Raises:
            InvalidInputFile: File is invalid
        """

        if not os.path.isfile(self.filename) or not self.filename.endswith(
                ".mzML"
        ):
            raise _InvalidInputFile(f"File {self.filename} is not valid!")

    def parse_file(self) -> Dict:
        """Reads the file line by line and obtains all information

        Data is then bulk processed by MS level

        Returns:
            Dict: Dictionary of each spectrum split by MS level
        """

        # CHeck the file exists and is an MzML file
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
        """Creates threads for processing MS1 and MS2 data simultaneously

        Arguments:
            ms_levels (List[_Spectrum]): Collection of MS spectra
        """

        pool = [
            Thread(target=self.process_spectra, args=(ms,)) for ms in ms_levels
        ]

        [thread.start() for thread in pool]
        [thread.join() for thread in pool]

    def process_spectra(self, spectra: List[_Spectrum]):
        """Processes spectra from a list and serialises the data

        Arguments:
            spectra (List[_Spectrum]): List of Spectra
        """

        for spec in spectra:
            spec.process()
            self.ms[spec.ms_level].append(spec)

    def build_output(self) -> Dict:
        """Builds the MS data output from the MS1 and MS2 data

        Returns:
            Dict: MS spectra split by level
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
                    spec.process()
                if spec.serialized["mass_list"]:
                    output["ms" + ms_level][
                        f"spectrum_{pos + 1}"
                    ] = spec.serialized

        return output

    def write_out_to_file(self):
        """Writes out the MS1 and MS2 data to JSON format

        If any spectra are not processed, they are processed here

        Arguments:
            ms1 List[Spectrum] MS1 spectra
            ms2 List[Spectrum] MS2 spectra
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
        """Processes a line from mzML

        Checks if we are in a spectrum or not
        If we're not in a spectrum, check for spectrum tag and pull information

        Continuously check if we've reached the end tag of the spectrum and
        add the spectrum to a list

        If we're in a spectrum and not reached the end tag,
        check and pull relevant information

        Arguments:
            line {str} -- Line form mzML
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
        """Initiates the spectrum data gathering process

        Check we get a match for spectrum index tag
        If not, we've got junk and just return
        If we are, extract all information from that line

        Arguments:
            line (str): Line from mzML
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
        """Attempts to extract information from a given line

        Information here:
        Retention Time
        32 or 64 bit data
        Type of compression
        MZ data
        Intensity Data

        Arguments:
            line (str): Line from mzML

        Raises:
            Exception: Unable to determine what kind of binary data
            we're looking at.
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
        """Updates the parent for MS3 and above

        Arguments:
            filter_string (str): String containing parent
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
    return _MzmlParser(
        filename,
        out_dir,
        rt_units=rt_units,
        int_threshold=int_threshold,
        relative_intensity=relative,
    ).parse_file()
