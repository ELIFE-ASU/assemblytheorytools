import base64
import json
import struct
import zlib

import pytest

from assemblytheorytools.tools_mzml import (
    _MzmlParser,
    _Spectrum,
    _UnsupportedCompressionMethod,
    process_mzml_file,
)


def _make_spectrum(
    mz, intensity, ms_level="1", scan="1", retention_time="1.5", precision=64
):
    """Build the encoded spectrum state produced by the XML parser."""
    spec = _Spectrum(intensity_threshold=0)
    spec.array_length = len(mz)
    spec.d_type = f"{precision}-bit float"
    spec.compression = "zlib compression"
    spec.ms_level = ms_level
    spec.scan = scan
    spec.retention_time = retention_time

    data_type = "f" if precision == 32 else "d"
    mz_bytes = struct.pack(f"<{len(mz)}{data_type}", *mz)
    intensity_bytes = struct.pack(f"<{len(intensity)}{data_type}", *intensity)
    spec.mz = base64.b64encode(zlib.compress(mz_bytes)).decode()
    spec.intensity = base64.b64encode(zlib.compress(intensity_bytes)).decode()
    return spec


def test_build_output_processes_unserialized_spectra():
    """The fallback must call process() before assembling the output."""
    spec = _make_spectrum(mz=[100.1234, 200.5678], intensity=[50000.0, 200.0])
    assert spec.serialized == {}

    parser = _MzmlParser.__new__(_MzmlParser)
    parser.ms = {"1": [spec]}

    output = parser.build_output()

    assert output == {"ms1": {"spectrum_1": spec.serialized}}
    assert spec.serialized["mass_list"] == [100.1234, 200.5678]


@pytest.mark.parametrize("precision", [32, 64])
@pytest.mark.parametrize(
    "ms_level, expected_peaks",
    [
        ("0", {}),
        ("1", {"300.5000": 1000}),
        ("2", {"200.2500": 1000, "300.5000": 1000}),
    ],
)
def test_process_spectrum_preserves_precision_and_strict_thresholds(
    precision, ms_level, expected_peaks
):
    spec = _make_spectrum(
        [100.125, 200.25, 300.5],
        [50, 1000, 1000.75],
        ms_level=ms_level,
        precision=precision,
    )
    spec.intensity_threshold = 1000

    spec.process()

    assert spec.mz == [100.125, 200.25, 300.5]
    assert spec.intensity == [50, 1000, 1000.75]
    assert spec.serialized == {
        **expected_peaks,
        "retention_time": "1.5",
        "scan": "1",
        "hcd": "",
        "mass_list": [float(mass) for mass in expected_peaks],
    }


def test_spectrum_keeps_duplicate_rounded_masses_and_last_intensity():
    spec = _make_spectrum([100.123411, 100.123422], [1100.9, 1200.5])

    spec.process()

    assert spec.serialized["100.1234"] == 1200
    assert spec.serialized["mass_list"] == [100.1234, 100.1234]


def test_relative_intensities_preserve_peak_order_and_base_peak():
    spec = _make_spectrum([100, 200, 300], [10.9, 50.9, 25.9])
    spec.relative = True

    spec.process()

    assert spec.serialized == {
        "retention_time": "1.5",
        "scan": "1",
        "hcd": "",
        "mass_list": [100.0, 200.0, 300.0],
        100.0: 20.0,
        300.0: 50.0,
        200.0: 100.0,
        "base_peak": [200.0, 50.0],
    }
    assert list(spec.serialized) == [
        "retention_time",
        "scan",
        "hcd",
        "mass_list",
        100.0,
        300.0,
        200.0,
        "base_peak",
    ]


def test_unsupported_compression_preserves_error_and_decoded_state():
    spec = _make_spectrum([100], [1000])
    spec.compression = "no compression"
    encoded_mz, encoded_intensity = spec.mz, spec.intensity

    with pytest.raises(
        _UnsupportedCompressionMethod,
        match=r"^Compression method no compression is not supported\.$",
    ):
        spec.process()

    assert spec.mz == base64.b64decode(encoded_mz)
    assert spec.intensity == base64.b64decode(encoded_intensity)
    assert spec.serialized == {}


def test_build_output_preserves_string_sorting_and_empty_spectrum_numbering():
    parser = _MzmlParser.__new__(_MzmlParser)
    parser.ms = {
        "1": [
            _make_spectrum([100], [500], scan="2", retention_time="2.0"),
            _make_spectrum([100], [500], scan="10", retention_time="10.0"),
            _make_spectrum([100], [0], scan="1", retention_time="1.0"),
        ]
    }

    spectra = parser.build_output()["ms1"]

    assert list(spectra) == ["spectrum_2", "spectrum_3"]
    assert [spectrum["scan"] for spectrum in spectra.values()] == ["10", "2"]
    assert [spectrum.scan for spectrum in parser.ms["1"]] == ["1", "10", "2"]


def test_process_mzml_file_extracts_metadata_and_writes_json(tmp_path):
    spec = _make_spectrum([100.125, 200.25], [50, 1000.75])
    source = tmp_path / "sample.mzML"
    source.write_text(
        f"""<mzML>
<run><spectrumList count="1">
<spectrum index="0" defaultArrayLength="2">
<cvParam accession="MS:1000511" value="2"/>
<cvParam accession="MS:1000796" value="scan=7"/>
<cvParam accession="MS:1000016" value="90"/>
<cvParam accession="MS:1000512" value="500@hcd35.00 fragment"/>
<userParam accession="MS:1000511" value="9"/>
<precursorList count="1">
<precursor spectrumRef="scan=6">
<cvParam accession="MS:1000744" value="500.123456"/>
</precursor></precursorList>
<binaryDataArrayList count="2">
<binaryDataArray>
<cvParam accession="MS:1000523" name="64-bit float"/>
<cvParam accession="MS:1000574" name="zlib compression"/>
<cvParam accession="MS:1000514" name="m/z array"/>
<binary>{spec.mz}</binary>
</binaryDataArray>
<binaryDataArray>
<cvParam accession="MS:1000515" name="intensity array"/>
<binary>{spec.intensity}</binary>
</binaryDataArray></binaryDataArrayList>
</spectrum>
</spectrumList></run>
</mzML>
"""
    )
    output_dir = tmp_path / "results" / "spectra"

    output = process_mzml_file(str(source), str(output_dir), rt_units="sec")

    assert output == {
        "ms2": {
            "spectrum_1": {
                "200.2500": 1000,
                "retention_time": "1.5",
                "scan": "7",
                "hcd": "35.00",
                "parent": "500.1235",
                "precursors": ["500.123456"],
                "parent_scan": "6",
                "precursors_scans": ["6"],
                "HCD": "35.00",
                "mass_list": [200.25],
            }
        }
    }
    assert json.loads((output_dir / "ripper_sample.json").read_text()) == output
