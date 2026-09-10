"""Opt-in checks against native C++ binaries with optional build features.

Run with ``--run-integration`` and set any applicable executable paths:
``ATT_TEST_CPP_PARALLEL_PATH``, ``ATT_TEST_CPP_TELEMETRY_PATH``,
``ATT_TEST_CPP_PARALLEL_TELEMETRY_PATH``, and ``ATT_TEST_CPP_PATH`` (ordinary
build without telemetry). Missing feature binaries skip their own tests.
"""

import json
import os
import shutil
from pathlib import Path

import pytest

import assemblytheorytools as att
from assemblytheorytools import assembly

pytestmark = pytest.mark.integration
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


def _binary(variable):
    configured = os.environ.get(variable)
    if not configured:
        pytest.skip(f"set {variable} to test this C++ build feature")
    executable = shutil.which(os.path.expanduser(configured))
    assert executable is not None, f"{variable} is not an executable: {configured}"
    return str(Path(executable).resolve())


@pytest.fixture(autouse=True)
def isolate_native_run_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(assembly.tempfile, "tempdir", str(tmp_path))


def _calculate(binary, **options):
    result = att.calculate_assembly_index(
        att.smi_to_nx(CAFFEINE), strip_hydrogen=True,
        dir_code=binary, cpp_options=att.AssemblyCppOptions(**options),
        return_log_file=True, timeout=30,
    )
    assert result[0] == 9
    assert result[1]
    assert result[2].number_of_edges() > 0
    log = Path(result[3])
    assert log.is_file()
    return log.parent


def test_native_forced_parallel_search_preserves_index_and_pathway():
    _calculate(_binary("ATT_TEST_CPP_PARALLEL_PATH"), parallel="on", threads=2)


def test_native_telemetry_is_retained_beside_returned_log():
    directory = _calculate(_binary("ATT_TEST_CPP_TELEMETRY_PATH"), telemetry=True)
    telemetry = json.loads((directory / "graph_inTelemetry.json").read_text())
    assert telemetry["schema_version"] >= 1
    assert telemetry["processed_graph"]["atoms"] > 0
    assert telemetry["counters"]["canonicalisation_calls"] > 0


def test_native_parallel_telemetry_confirms_requested_thread_count():
    directory = _calculate(
        _binary("ATT_TEST_CPP_PARALLEL_TELEMETRY_PATH"),
        parallel="on", threads=2, telemetry=True,
    )
    telemetry = json.loads((directory / "graph_inTelemetry.json").read_text())
    parallel = telemetry["parallel"]
    assert parallel["enabled"]
    assert parallel["local_threads"] == 2
    assert len(parallel["workers"]) == parallel["worker_count"]
    assert {worker["local_worker_index"] for worker in parallel["workers"]} == {0, 1}


def test_native_ordinary_build_reports_unsupported_telemetry():
    binary = _binary("ATT_TEST_CPP_PATH")
    with pytest.raises(OSError, match="unknown option.*telemetry") as error:
        _calculate(binary, telemetry=True)
    assert "AssemblyCpp exited with status 2" in str(error.value)
    assert "AssemblyCpp log:" in str(error.value)
