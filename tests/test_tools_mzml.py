import base64
import struct
import zlib

from assemblytheorytools.tools_mzml import _MzmlParser, _Spectrum


def _make_spectrum(mz, intensity, ms_level="1", scan="1", retention_time="1.5"):
    """
    Build a raw, un-processed _Spectrum with base64/zlib-encoded binary data.

    Parameters
    ----------
    mz : list of float
        m/z values for the spectrum's peaks.
    intensity : list of float
        Intensity values, one per m/z value.
    ms_level : str, optional
        MS level string, by default "1".
    scan : str, optional
        Scan number string, by default "1".
    retention_time : str, optional
        Retention time string, by default "1.5".

    Returns
    -------
    _Spectrum
        A spectrum with encoded (but not yet decoded/serialized) mz and
        intensity data, matching the state _MzmlParser leaves a spectrum in
        immediately after parsing its XML but before `.process()` runs.
    """
    spec = _Spectrum(intensity_threshold=0)
    spec.array_length = len(mz)
    spec.d_type = "64-bit float"
    spec.compression = "zlib compression"
    spec.ms_level = ms_level
    spec.scan = scan
    spec.retention_time = retention_time

    mz_bytes = struct.pack(f"<{len(mz)}d", *mz)
    intensity_bytes = struct.pack(f"<{len(intensity)}d", *intensity)
    spec.mz = base64.b64encode(zlib.compress(mz_bytes)).decode()
    spec.intensity = base64.b64encode(zlib.compress(intensity_bytes)).decode()
    return spec


def test_build_output_processes_unserialized_spectra():
    """
    Regression test: build_output must process spectra that haven't been
    serialized yet, not crash on them.

    _MzmlParser.build_output has a fallback for spectra whose `.serialized`
    is still empty (the normal parse flow always processes spectra before
    build_output runs, so this branch is rarely exercised) that used to call
    a nonexistent `spec.rma_process()` instead of `spec.process()`, which
    would raise AttributeError if it were ever reached.

    Asserts:
        - build_output does not raise.
        - The unserialized spectrum's mass_list ends up populated in the
          output, proving `.process()` actually ran.
    """
    spec = _make_spectrum(mz=[100.1234, 200.5678], intensity=[50000.0, 200.0])
    assert not spec.serialized  # sanity check: this is the "not yet processed" case

    parser = _MzmlParser.__new__(_MzmlParser)
    parser.ms = {"1": [spec]}

    output = parser.build_output()

    assert spec.serialized  # process() was called and populated it
    assert output["ms1"]["spectrum_1"]["mass_list"]
