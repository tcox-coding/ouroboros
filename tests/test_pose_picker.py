"""pose_picker: the LLM picks a saved pose from a menu (or none); bad answers mean none."""
from ouroboros import pose_picker as pp

POSES = [{"name": "hands_on_hips", "description": "standing, hands on hips, cowboy shot"},
         {"name": "sitting_cross_legged", "description": "sitting, cross-legged, full body"}]


class Backend:
    def __init__(self, answer):
        self.answer, self.parts, self.schema = answer, None, None

    def complete(self, instructions, parts, schema, name, max_side):
        self.parts, self.schema = parts, schema
        return self.answer, 0.002, 50


def test_a_listed_pose_is_picked_and_the_menu_is_the_only_choice():
    b = Backend({"pose": "sitting_cross_legged", "reason": "asks for sitting"})
    out = pp.pick_pose(b, POSES, description="a girl sitting on the floor")
    assert out["pose"] == "sitting_cross_legged" and out["cost"] == 0.002
    assert b.schema["properties"]["pose"]["enum"] == ["hands_on_hips", "sitting_cross_legged", "none"]
    assert "hands_on_hips: standing, hands on hips" in b.parts[0]["text"]


def test_none_or_an_unknown_name_means_no_pose():
    assert pp.pick_pose(Backend({"pose": "none", "reason": "running"}), POSES, description="running")["pose"] is None
    assert pp.pick_pose(Backend({"pose": "made_up", "reason": ""}), POSES, description="x")["pose"] is None


def test_an_empty_library_skips_the_call():
    assert pp.pick_pose(None, [], description="x") == {"pose": None, "reason": "the pose library is empty",
                                                        "cost": 0.0, "considered": 0}


def test_the_reference_is_shown_only_without_a_description_or_prompt():
    b = Backend({"pose": "none", "reason": ""})
    pp.pick_pose(b, POSES, description="sitting", reference="IMG")
    assert not any("image" in p for p in b.parts)
    pp.pick_pose(b, POSES, reference="IMG")
    assert any(p.get("image") == "IMG" for p in b.parts)


def test_large_libraries_are_shortlisted_by_shared_words():
    many = [{"name": f"pose_{i}", "description": "standing"} for i in range(60)] + \
           [{"name": "kneeling_prayer", "description": "kneeling, hands together"}]
    menu = pp.shortlist(many, "a knight kneeling with hands together")
    assert len(menu) == pp.MAX_MENU and menu[0]["name"] == "kneeling_prayer"
