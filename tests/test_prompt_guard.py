"""Prompt Guard logic tests that do not need the model: segmentation, threshold decision, scoring rows."""
import pytest

from defenses.prompt_guard import decide, split_segments
from run_detector import score_case


def test_short_text_is_one_segment():
    assert split_segments(list(range(10)), 510, 255) == [list(range(10))]
    assert split_segments([], 510, 255) == [[]]


def test_long_text_overlaps_and_covers_everything():
    ids = list(range(1200))
    segs = split_segments(ids, 510, 255)
    assert all(len(s) <= 510 for s in segs)
    assert segs[0][:3] == [0, 1, 2] and segs[-1][-1] == 1199
    covered = set()
    for s in segs:
        covered.update(s)
    assert covered == set(ids)
    assert segs[1][0] == 255  # 50% overlap


def test_exact_window_is_one_segment_and_one_over_is_two():
    assert len(split_segments(list(range(510)), 510, 255)) == 1
    assert len(split_segments(list(range(511)), 510, 255)) == 2


def test_bad_segment_arguments_raise():
    for content_len, stride in ((0, 1), (10, 0), (10, 11)):
        with pytest.raises(ValueError):
            split_segments([1, 2, 3], content_len, stride)


def test_decide_threshold_is_inclusive():
    assert decide(0.5, 0.5) == "block"
    assert decide(0.4999, 0.5) == "allow"


class FakeDetector:
    device = "cpu"

    def __init__(self, scores):
        self.scores = scores

    def score_text(self, text):
        return self.scores[text], 1, 0.01


RECORD = {"case_id": "X-ATTACK", "pair_id": "X", "condition": "attack", "split": "dev",
          "delivery": "indirect_document", "objective": "attacker_response", "technique": "t",
          "untrusted_content": "doc", "user_prompt": "usr"}


def test_case_score_is_the_max_of_the_two_boundaries():
    row = score_case(FakeDetector({"doc": 0.9, "usr": 0.1}), RECORD)
    assert row["score"] == 0.9 and row["max_boundary"] == "document"
    row = score_case(FakeDetector({"doc": 0.2, "usr": 0.7}), RECORD)
    assert row["score"] == 0.7 and row["max_boundary"] == "user_prompt"
    assert row["doc_score"] == 0.2 and row["user_score"] == 0.7
