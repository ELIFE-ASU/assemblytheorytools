"""Keep the Rust wrappers' public error contract consistent."""

import pytest

import assemblytheorytools.assembly as assembly


@pytest.mark.parametrize(
    "wrapper, backend_method",
    [
        (assembly.calculate_assembly_index_rust, "index"),
        (assembly.calculate_assembly_depth_rust, "depth"),
        (assembly.get_molecule_info_rust, "mol_info"),
    ],
)
@pytest.mark.parametrize("failure_stage", ["conversion", "backend"])
def test_rust_wrappers_translate_os_errors(
    monkeypatch, wrapper, backend_method, failure_stage
):
    error = OSError("unreadable mol block")

    def fail(*args, **kwargs):
        raise error

    if failure_stage == "conversion":
        monkeypatch.setattr(assembly, "_mol_to_molblock", fail)
    else:
        monkeypatch.setattr(assembly.at_rust, backend_method, fail)

    with pytest.raises(ValueError) as raised:
        wrapper(assembly.Chem.MolFromSmiles("CCO"))

    assert str(raised.value) == (
        "The Rust backend could not read this molecule: unreadable mol block"
    )
    assert raised.value.__cause__ is error
