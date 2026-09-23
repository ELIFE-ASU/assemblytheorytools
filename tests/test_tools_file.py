import json
import os

import pytest
from filelock import FileLock, Timeout

import assemblytheorytools as att
from assemblytheorytools import tools_file
from assemblytheorytools.tools_file import prep_json


def test_file_list_returns_only_direct_files(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.dat").write_text("b")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.txt").write_text("c")

    assert set(att.file_list(tmp_path)) == {"a.txt", "b.dat"}

    monkeypatch.chdir(tmp_path)
    assert set(att.file_list()) == {"a.txt", "b.dat"}
    assert set(att.file_list("")) == {"a.txt", "b.dat"}


def test_file_list_all_recurses_and_preserves_path_style(tmp_path, monkeypatch):
    nested = tmp_path / "one" / "two"
    nested.mkdir(parents=True)
    direct_file = tmp_path / "direct.txt"
    nested_file = nested / "nested.txt"
    direct_file.write_text("direct")
    nested_file.write_text("nested")

    assert set(att.file_list_all(tmp_path)) == {str(direct_file), str(nested_file)}

    monkeypatch.chdir(tmp_path)
    # os.walk joins with the platform separator, so the expectation has to too.
    assert set(att.file_list_all(".")) == {
        os.path.join(".", "direct.txt"),
        os.path.join(".", "one", "two", "nested.txt"),
    }
    assert set(att.file_list_all("")) == {str(direct_file), str(nested_file)}


def test_filter_files_checks_basenames_and_preserves_iterable_order():
    paths = [
        "/matching-directory/result.csv",
        "/data/matching-result.txt",
        "/data/other.txt",
        "/data/matching-result.csv",
        "/data/matching-result.txt",
    ]

    assert att.filter_files(iter(paths), "matching") == [
        "/data/matching-result.txt",
        "/data/matching-result.csv",
        "/data/matching-result.txt",
    ]


def test_write_to_shared_file_appends_without_adding_content(tmp_path):
    shared_file = tmp_path / "shared.log"

    att.write_to_shared_file("first\n", shared_file)
    att.write_to_shared_file("second", shared_file)

    assert shared_file.read_text() == "first\nsecond"


def test_write_to_shared_file_holds_lock_through_buffered_writes(tmp_path, monkeypatch):
    """The stream is closed inside the lock, so buffered bytes land before it lifts."""
    shared_file = tmp_path / "shared.log"
    events = []

    class RecordingLock:
        def __init__(self, lock_file):
            events.append(("requested", os.path.basename(lock_file)))

        def __enter__(self):
            events.append(("acquired", None))
            return self

        def __exit__(self, *exception):
            # Reading here is what proves the buffer already reached disk.
            events.append(("released", shared_file.read_text()))
            return False

    monkeypatch.setattr(tools_file, "FileLock", RecordingLock)
    att.write_to_shared_file("buffered message", shared_file)

    assert events == [
        ("requested", "shared.log.lock"),
        ("acquired", None),
        ("released", "buffered message"),
    ]


def test_write_to_shared_file_excludes_a_concurrent_writer(tmp_path):
    """A held sidecar lock is what makes a second writer wait its turn."""
    shared_file = tmp_path / "shared.log"
    lock_file = f"{shared_file}.lock"

    with FileLock(lock_file):
        with pytest.raises(Timeout):
            # A distinct instance contends through the operating system, which
            # is the same path a second process takes.
            FileLock(lock_file, timeout=0).acquire()

    att.write_to_shared_file("released\n", shared_file)
    assert shared_file.read_text() == "released\n"


def test_remove_files_removes_nested_files_but_preserves_directories(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "top.txt").write_text("top")
    (nested / "child.txt").write_text("child")

    att.remove_files(tmp_path)

    assert list(tmp_path.iterdir()) == [nested]
    assert list(nested.iterdir()) == []


@pytest.mark.parametrize("remove_directory", [att.safe_folder_remove, att.wipe_dir])
def test_directory_removal_deletes_a_nested_tree(tmp_path, remove_directory):
    target = tmp_path / "target"
    nested = target / "nested" / "deeper"
    nested.mkdir(parents=True)
    (target / "top.txt").write_text("top")
    (nested / "child.txt").write_text("child")

    remove_directory(target)

    assert not target.exists()


def test_list_subdirs_filters_by_prefix(tmp_path):
    (tmp_path / "ai_calc_1").mkdir()
    (tmp_path / "ai_calc_2").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "ai_calc_file").write_text("not a directory")

    assert set(att.list_subdirs(tmp_path)) == {"ai_calc_1", "ai_calc_2"}
    assert att.list_subdirs(tmp_path, target="other") == ["other"]


@pytest.mark.parametrize(
    ("colours", "expected"),
    [
        pytest.param(
            'red, , "blue", 3', ["red", "ERROR", "blue", "3"], id="mixed-types"
        ),
        pytest.param("", ["ERROR"], id="empty-list"),
        pytest.param('"blue", ', ["blue", "ERROR"], id="trailing-entry"),
        pytest.param("\nred,\n, blue\n", ["red", "ERROR", "blue"], id="multiline"),
    ],
)
def test_prep_json_repairs_edge_colours_without_changing_other_fields(
    tmp_path, colours, expected
):
    path = tmp_path / "pathway.json"
    path.write_text('{"EdgeColours": [' + colours + '], "unchanged": [1, 2]}')

    prep_json(path)

    assert json.loads(path.read_text()) == {
        "EdgeColours": expected,
        "unchanged": [1, 2],
    }


def test_prep_json_leaves_invalid_json_unchanged(tmp_path):
    path = tmp_path / "pathway.json"
    original = '{"EdgeColours": [red,], "other": invalid}'
    path.write_text(original)

    with pytest.raises(json.JSONDecodeError):
        prep_json(path)

    assert path.read_text() == original


def test_remove_file_pattern_removes_only_matching_files(tmp_path):
    matching = [tmp_path / "one.tmp", tmp_path / "two.tmp"]
    for path in matching:
        path.write_text("remove")
    keep = tmp_path / "keep.txt"
    keep.write_text("keep")
    matching_directory = tmp_path / "directory.tmp"
    matching_directory.mkdir()

    att.remove_file_pattern(str(tmp_path / "*.tmp"))

    assert all(not path.exists() for path in matching)
    assert keep.read_text() == "keep"
    assert matching_directory.is_dir()


def test_safe_folder_remove_ignores_missing_directories(tmp_path):
    target = tmp_path / "missing"

    att.safe_folder_remove(target)

    assert not target.exists()


@pytest.mark.parametrize("remove_directory", [att.safe_folder_remove, att.wipe_dir])
def test_directory_removal_ignores_regular_files(tmp_path, remove_directory):
    path = tmp_path / "keep.txt"
    path.write_text("keep")

    remove_directory(path)

    assert path.read_text() == "keep"
