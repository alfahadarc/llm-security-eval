"""Threshold selection tests on hand-made score rows."""
import pytest

from choose_threshold import choose, rates


def rows(attack, other):
    return ([{"condition": "attack", "score": s} for s in attack]
            + [{"condition": "clean", "score": s} for s in other])


def test_rates_use_inclusive_threshold():
    r = rows([0.9, 0.5], [0.5, 0.1])
    assert rates(r, 0.5) == (1.0, 0.5)
    assert rates(r, 0.6) == (0.5, 0.0)


def test_perfect_separation_picks_the_highest_threshold_that_keeps_full_recall():
    threshold, recall, fpr, _ = choose(rows([0.9, 0.8, 0.7], [0.1, 0.2, 0.3]))
    assert (threshold, recall, fpr) == (0.7, 1.0, 0.0)


def test_overlap_trades_recall_for_fpr_by_youden_j():
    # at 0.4: recall 3/3, fpr 1/3 -> J .667 ; at 0.7: recall 2/3, fpr 0 -> J .667 ; tie goes to the higher threshold
    threshold, recall, fpr, _ = choose(rows([0.95, 0.7, 0.4], [0.4, 0.2, 0.1]))
    assert threshold == 0.7 and recall == pytest.approx(2 / 3) and fpr == 0


def test_needs_both_classes():
    with pytest.raises(ValueError):
        choose(rows([0.9], []))
    with pytest.raises(ValueError):
        choose(rows([], [0.1]))


def test_only_dev_rows_are_used_by_main(tmp_path, monkeypatch, capsys):
    import json
    import choose_threshold
    scores = tmp_path / "s.jsonl"
    data = ([{"split": "dev", "condition": "attack", "score": 0.9}, {"split": "dev", "condition": "clean", "score": 0.1}]
            + [{"split": "test", "condition": "attack", "score": 0.01}, {"split": "test", "condition": "clean", "score": 0.99}])
    scores.write_text("\n".join(json.dumps(r) for r in data) + "\n")
    monkeypatch.setattr("sys.argv", ["choose_threshold.py", "--scores", str(scores)])
    assert choose_threshold.main() == 0
    assert "dev cases: 2" in capsys.readouterr().out
