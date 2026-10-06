"""Folders and favourites for the saved poses, styles and characters (organize.py)."""

from PIL import Image

from ouroboros import organize, reflib


def test_folder_names_are_cleaned():
    assert organize.clean_folder(" anime / female\\ok ") == "anime/female/ok"
    assert organize.clean_folder("../_removed/.hidden/x") == "x"
    assert organize.clean_folder(None) == "" and organize.clean_folder("///") == ""


def test_items_get_their_folder_and_favourite_and_every_parent_folder_is_listed(tmp_path):
    organize.update(tmp_path, ["a", "b"], folder="anime/female")
    organize.update(tmp_path, ["b"], favorite=True)
    organize.add_folder(tmp_path, "empty")
    out = organize.annotate(tmp_path, [{"name": "a"}, {"name": "b"}, {"name": "c"}])
    assert [(i["name"], i["folder"], i["favorite"]) for i in out["items"]] == [
        ("a", "anime/female", False), ("b", "anime/female", True), ("c", "", False)]
    assert out["folders"] == ["anime", "anime/female", "empty"]


def test_a_folder_stays_when_its_items_move_out_or_are_deleted(tmp_path):
    organize.update(tmp_path, ["a"], folder="x")
    organize.update(tmp_path, ["a"], folder="")
    organize.update(tmp_path, ["b"], folder="y")
    organize.forget(tmp_path, "b")
    out = organize.annotate(tmp_path, [{"name": "a"}])
    assert out["items"][0]["folder"] == "" and out["folders"] == ["x", "y"]
    assert "b" not in organize.load(tmp_path)["items"]


def test_renaming_a_folder_moves_its_items_and_subfolders_and_removing_moves_them_up(tmp_path):
    organize.update(tmp_path, ["a"], folder="old")
    organize.update(tmp_path, ["b"], folder="old/sub")
    organize.update(tmp_path, ["c"], folder="older")  # a prefix, not inside "old"
    organize.move_folder(tmp_path, "old", "new/place")
    out = organize.annotate(tmp_path, [{"name": n} for n in "abc"])
    assert [i["folder"] for i in out["items"]] == ["new/place", "new/place/sub", "older"]
    organize.move_folder(tmp_path, "new/place", "new")  # remove: contents up a level
    out = organize.annotate(tmp_path, [{"name": n} for n in "abc"])
    assert [i["folder"] for i in out["items"]] == ["new", "new/sub", "older"]
    assert "new/place" not in out["folders"]


def test_a_folder_cant_go_inside_itself(tmp_path):
    import pytest
    organize.update(tmp_path, ["a"], folder="x")
    with pytest.raises(ValueError):
        organize.move_folder(tmp_path, "x", "x/y")


def test_the_library_file_doesnt_show_up_as_an_item(tmp_path):
    lib = reflib.RefLibrary(tmp_path, "style")
    lib.add("ink", Image.new("RGB", (8, 8)))
    organize.update(lib.root, ["ink"], folder="wash", favorite=True)
    assert [p["name"] for p in lib.list()] == ["ink"]
    lib.remove("ink")
    assert lib.list() == []


def test_the_server_files_saved_items_and_lists_them_with_their_folders(tmp_path, monkeypatch):
    from ouroboros import runner, server
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)  # rel_url
    server.ref_library("character").add("knight", Image.new("RGB", (8, 8)))
    out = server.organize_library("character", {"names": ["knight"], "folder": "heroes", "favorite": True})
    assert out["folders"] == ["heroes"]
    assert (out["items"][0]["folder"], out["items"][0]["favorite"]) == ("heroes", True)
    assert server.organize_library("character", {"new_folder": "villains"})["folders"] == ["heroes", "villains"]
    out = server.organize_library("character", {"move_folder": "heroes", "to": "good/heroes"})
    assert out["items"][0]["folder"] == "good/heroes" and "heroes" not in out["folders"]
    assert server.poses_list()["folders"] == []  # each library keeps its own
