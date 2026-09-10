"""Validate the C++ option contract independently of executable availability."""

from dataclasses import FrozenInstanceError

import pytest

from assemblytheorytools._cpp_options import AssemblyCppOptions

UINT64_MAX = (1 << 64) - 1
INT_MAX = (1 << 31) - 1
BASE_ARGUMENTS = ["-removeHydrogens=0", "-compensateDisjoint=0", "-memTest=0"]


@pytest.mark.parametrize("str_mode", [False, True])
def test_defaults_preserve_legacy_command_line(str_mode):
    options = AssemblyCppOptions()
    assert options._arguments(str_mode=str_mode) == BASE_ARGUMENTS + (
        ["-runStrings=1"] if str_mode else []
    )
    assert not options._retain_files


def test_options_are_immutable():
    options = AssemblyCppOptions()
    with pytest.raises(FrozenInstanceError):
        options.pathway = False


@pytest.mark.parametrize(
    "settings, flag, str_mode",
    [
        ({"runtime_ticks": 0}, "-runTime=0", False),
        ({"runtime_ticks": UINT64_MAX}, f"-runTime={UINT64_MAX}", True),
        ({"enum_max": 1}, "-enumMax=1", False),
        ({"enum_max": INT_MAX}, f"-enumMax={INT_MAX}", False),
        ({"pathway": False}, "--pathway=0", False),
        ({"pathway": False}, "--pathway=0", True),
        ({"accept_palindromes": True}, "-acceptPalindromes=1", True),
        ({"parallel": "auto"}, "--parallel=auto", False),
        ({"parallel": "auto"}, "--parallel=auto", True),
        ({"parallel": "on"}, "--parallel=on", False),
        ({"threads": 1}, "--threads=1", False),
        ({"threads": INT_MAX}, f"--threads={INT_MAX}", False),
        ({"verbose": True}, "--verbose=1", False),
        ({"memory_report": True}, "-memTest=1", False),
        ({"memory_report": True}, "-memTest=1", True),
        ({"telemetry": True}, "--telemetry=1", False),
        ({"write_intermediate_mas": True}, "-writeIntermediateMAs=1", False),
    ],
)
def test_each_control_maps_to_a_unique_flag(settings, flag, str_mode):
    arguments = AssemblyCppOptions(**settings)._arguments(str_mode=str_mode)
    assert flag in arguments
    names = [argument.split("=", 1)[0] for argument in arguments]
    assert len(names) == len(set(names))
    assert "-removeHydrogens=0" in arguments
    assert "-compensateDisjoint=0" in arguments


@pytest.mark.parametrize("name", ["runtime_ticks", "enum_max", "threads"])
@pytest.mark.parametrize("value", [True, False, 1.0, [], {}])
def test_numeric_controls_reject_non_integer_types(name, value):
    with pytest.raises(TypeError, match=name):
        AssemblyCppOptions(**{name: value})


@pytest.mark.parametrize("name", ["runtime_ticks", "enum_max"])
def test_numeric_budget_controls_reject_strings(name):
    with pytest.raises(TypeError, match=name):
        AssemblyCppOptions(**{name: "1"})


@pytest.mark.parametrize(
    "name, value",
    [
        ("runtime_ticks", -1), ("runtime_ticks", UINT64_MAX + 1),
        ("enum_max", 0), ("enum_max", -1), ("enum_max", INT_MAX + 1),
        ("threads", 0), ("threads", -1), ("threads", INT_MAX + 1),
    ],
)
def test_numeric_controls_reject_values_outside_cpp_ranges(name, value):
    with pytest.raises(ValueError, match=name):
        AssemblyCppOptions(**{name: value})


@pytest.mark.parametrize(
    "name",
    ["pathway", "accept_palindromes", "verbose", "memory_report", "telemetry", "write_intermediate_mas"],
)
@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_boolean_controls_require_bool(name, value):
    with pytest.raises(TypeError, match=name):
        AssemblyCppOptions(**{name: value})


@pytest.mark.parametrize("value", ["ON", "", "yes", "0"])
def test_parallel_rejects_invalid_modes(value):
    with pytest.raises(ValueError, match="parallel"):
        AssemblyCppOptions(parallel=value)


@pytest.mark.parametrize("value", [True, 1, None, []])
def test_parallel_rejects_invalid_types(value):
    with pytest.raises(TypeError, match="parallel"):
        AssemblyCppOptions(parallel=value)


@pytest.mark.parametrize("value", ["AUTO", "1", "", "on"])
def test_threads_rejects_invalid_strings(value):
    with pytest.raises(ValueError, match="threads"):
        AssemblyCppOptions(threads=value)


def test_threads_rejects_none():
    with pytest.raises(TypeError, match="threads"):
        AssemblyCppOptions(threads=None)


@pytest.mark.parametrize("name", ["memory_report", "telemetry", "write_intermediate_mas"])
def test_diagnostic_output_requests_retain_files(name):
    assert AssemblyCppOptions(**{name: True})._retain_files


def test_pathway_and_verbose_do_not_request_additional_file_retention():
    assert not AssemblyCppOptions(pathway=True, verbose=True)._retain_files


def test_graph_mode_rejects_string_reversal_control():
    with pytest.raises(ValueError, match="accept_palindromes.*only.*string"):
        AssemblyCppOptions(accept_palindromes=True)._arguments(str_mode=False)


@pytest.mark.parametrize(
    "settings, name",
    [
        ({"parallel": "on"}, "parallel"),
        ({"enum_max": 50_000_000}, "enum_max"),
        ({"threads": 1}, "threads"),
        ({"verbose": True}, "verbose"),
        ({"telemetry": True}, "telemetry"),
        ({"write_intermediate_mas": True}, "write_intermediate_mas"),
    ],
)
def test_string_mode_rejects_unavailable_controls(settings, name):
    with pytest.raises(ValueError, match=f"{name}.*string"):
        AssemblyCppOptions(**settings)._arguments(str_mode=True)


@pytest.mark.parametrize("runtime_ticks", [0, 1, UINT64_MAX - 1])
def test_forced_parallel_rejects_finite_runtime(runtime_ticks):
    with pytest.raises(ValueError, match="parallel.*finite runtime_ticks"):
        AssemblyCppOptions(parallel="on", runtime_ticks=runtime_ticks)._arguments(str_mode=False)


@pytest.mark.parametrize("runtime_ticks", [None, UINT64_MAX])
def test_forced_parallel_accepts_unlimited_runtime(runtime_ticks):
    arguments = AssemblyCppOptions(parallel="on", runtime_ticks=runtime_ticks)._arguments(str_mode=False)
    assert "--parallel=on" in arguments


def test_forced_parallel_rejects_intermediate_output():
    with pytest.raises(ValueError, match="parallel.*write_intermediate_mas"):
        AssemblyCppOptions(parallel="on", write_intermediate_mas=True)._arguments(str_mode=False)


def test_auto_parallel_permits_cpp_serial_fallback():
    arguments = AssemblyCppOptions(
        parallel="auto", runtime_ticks=1, write_intermediate_mas=True,
    )._arguments(str_mode=False)
    assert "--parallel=auto" in arguments
    assert "-runTime=1" in arguments
    assert "-writeIntermediateMAs=1" in arguments


def test_all_graph_controls_serialize_together_without_duplicates():
    options = AssemblyCppOptions(
        runtime_ticks=100, enum_max=500, pathway=False, parallel="auto",
        threads=2, verbose=True, memory_report=True, telemetry=True,
        write_intermediate_mas=True,
    )
    assert options._arguments(str_mode=False) == [
        "-removeHydrogens=0", "-compensateDisjoint=0", "-memTest=1",
        "-runTime=100", "-enumMax=500", "--pathway=0", "--parallel=auto",
        "--threads=2", "--verbose=1", "--telemetry=1", "-writeIntermediateMAs=1",
    ]


def test_all_string_controls_serialize_together_without_duplicates():
    options = AssemblyCppOptions(
        runtime_ticks=100, pathway=False, accept_palindromes=True,
        parallel="auto", memory_report=True,
    )
    assert options._arguments(str_mode=True) == [
        "-removeHydrogens=0", "-compensateDisjoint=0", "-memTest=1",
        "-runStrings=1", "-runTime=100", "--pathway=0",
        "-acceptPalindromes=1", "--parallel=auto",
    ]
