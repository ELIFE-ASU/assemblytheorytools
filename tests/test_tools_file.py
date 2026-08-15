import json

import assemblytheorytools as att
from assemblytheorytools.tools_file import prep_json


def test_file_list_returns_only_direct_files(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.dat").write_text("b")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.txt").write_text("c")

    assert set(att.file_list(tmp_path)) == {"a.txt", "b.dat"}

    monkeypatch.chdir(tmp_path)
    assert set(att.file_list()) == {"a.txt", "b.dat"}


def test_file_list_all_recurses_and_returns_paths(tmp_path):
    nested = tmp_path / "one" / "two"
    nested.mkdir(parents=True)
    direct_file = tmp_path / "direct.txt"
    nested_file = nested / "nested.txt"
    direct_file.write_text("direct")
    nested_file.write_text("nested")

    assert set(att.file_list_all(tmp_path)) == {str(direct_file), str(nested_file)}


def test_filter_files_checks_the_basename_only():
    paths = [
        "/matching-directory/result.csv",
        "/data/matching-result.txt",
        "/data/other.txt",
    ]

    assert att.filter_files(paths, "matching") == ["/data/matching-result.txt"]


def test_write_to_shared_file_appends_without_adding_content(tmp_path):
    shared_file = tmp_path / "shared.log"

    att.write_to_shared_file("first\n", shared_file)
    att.write_to_shared_file("second", shared_file)

    assert shared_file.read_text() == "first\nsecond"


def test_remove_files_removes_nested_files_but_preserves_directories(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "top.txt").write_text("top")
    (nested / "child.txt").write_text("child")

    att.remove_files(tmp_path)

    assert list(tmp_path.iterdir()) == [nested]
    assert list(nested.iterdir()) == []


def test_wipe_dir_removes_a_nested_directory_tree(tmp_path):
    target = tmp_path / "target"
    nested = target / "nested" / "deeper"
    nested.mkdir(parents=True)
    (target / "top.txt").write_text("top")
    (nested / "child.txt").write_text("child")

    att.wipe_dir(target)

    assert not target.exists()


def test_list_subdirs_filters_by_prefix(tmp_path):
    (tmp_path / "ai_calc_1").mkdir()
    (tmp_path / "ai_calc_2").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "ai_calc_file").write_text("not a directory")

    assert set(att.list_subdirs(tmp_path)) == {"ai_calc_1", "ai_calc_2"}
    assert att.list_subdirs(tmp_path, target="other") == ["other"]


def test_prep_json_repairs_missing_and_unquoted_edge_colours(tmp_path):
    path = tmp_path / "pathway.json"
    path.write_text(
        '{"EdgeColours": [red, , "blue", 3], "unchanged": [1, 2]}'
    )

    prep_json(path)

    assert json.loads(path.read_text()) == {
        "EdgeColours": ["red", "ERROR", "blue", "3"],
        "unchanged": [1, 2],
    }


def test_remove_file_pattern_removes_only_matching_files(tmp_path):
    matching = [tmp_path / "one.tmp", tmp_path / "two.tmp"]
    for path in matching:
        path.write_text("remove")
    keep = tmp_path / "keep.txt"
    keep.write_text("keep")

    att.remove_file_pattern(str(tmp_path / "*.tmp"))

    assert all(not path.exists() for path in matching)
    assert keep.read_text() == "keep"


def test_safe_folder_remove_handles_nested_and_missing_directories(tmp_path):
    target = tmp_path / "target"
    nested = target / "nested"
    nested.mkdir(parents=True)
    (target / "top.txt").write_text("top")
    (nested / "child.txt").write_text("child")

    att.safe_folder_remove(target)
    att.safe_folder_remove(target)

    assert not target.exists()
