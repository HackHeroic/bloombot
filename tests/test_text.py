import numpy as np
import pytest

from bloombot.taxonomy import Level, parse_label
from bloombot.text.linguistics import analyze, combine_max
from bloombot.text.normalize import clean_question, fix_mojibake


@pytest.mark.parametrize("raw,scheme,expected", [
    ("Remember", None, Level.REMEMBER), ("analyse", None, Level.ANALYZE),
    ("Synthesis", None, Level.CREATE), ("Evaluation", None, Level.EVALUATE),
    ("BT5", "revised", Level.EVALUATE), ("BT5", "original", Level.CREATE),
    ("L6", "revised", Level.CREATE), (3, "revised", Level.APPLY), ("", None, None), ("banana", None, None),
])
def test_parse_label(raw, scheme, expected):
    assert parse_label(raw, scheme) == expected


def test_numeric_code_needs_scheme():
    with pytest.raises(ValueError):
        parse_label("BT5")


@pytest.mark.parametrize("raw,clean", [
    ("Q1. (a) Define a deadlock. [2 Marks] (CO1, BT1)", "Define a deadlock."),
    ("(ii) Compare TCP and UDP. CO3 BT4", "Compare TCP and UDP."),
    ("1.5 kg of sugar costs Rs 60. Find the cost of 4 kg.", "1.5 kg of sugar costs Rs 60. Find the cost of 4 kg."),
    ("Find x if x^2 = 16", "Find x if x^2 = 16"),
    ("Solve equation (2) for x.", "Solve equation (2) for x."),
    ("Apply L2 regularization to the model.", "Apply L2 regularization to the model."),
    ("Explain paging with a neat diagram. 10", "Explain paging with a neat diagram."),
])
def test_clean_question(raw, clean):
    assert clean_question(raw) == clean


def test_fix_mojibake():
    assert fix_mojibake("Newtonâ\u0080\u0099s law") == "Newton’s law"
    assert fix_mojibake("café") == "café"


def test_combine_max_is_distribution_of_maximum():
    a = np.array([1, 0, 0, 0, 0, 0.0])  # surely Remember
    b = np.array([0, 0.5, 0, 0, 0, 0.5])
    np.testing.assert_allclose(combine_max([a, b]), b)
    p = combine_max([np.full(6, 1 / 6), np.full(6, 1 / 6)])
    assert p.sum() == pytest.approx(1) and p[-1] > p[0]


@pytest.mark.parametrize("q,verbs,top", [
    ("List the types of operating systems.", ["list"], Level.REMEMBER),     # noun-tagged imperative
    ("Explain how you would design a database schema.", ["explain"], Level.UNDERSTAND),  # verb trap
    ("Define deadlock and explain its conditions.", ["define", "explain"], Level.UNDERSTAND),
    ("Write a program to reverse a linked list.", ["write"], Level.APPLY),
    ("Write short notes on virtual memory.", ["write"], Level.UNDERSTAND),
    ("Evaluate the integral of x squared from 0 to 1.", ["evaluate"], Level.APPLY),
    ("Come up with a new marketing strategy.", ["come"], Level.CREATE),
    ("Using Ohm's law, calculate the current.", ["calculate"], Level.APPLY),
    ("Do not just list the features; evaluate them.", ["evaluate"], Level.EVALUATE),
])
def test_directives(q, verbs, top):
    a = analyze(q)
    assert a.verbs == verbs
    assert Level(int(np.argmax(a.prior)) + 1) == top


def test_frames_and_masking():
    a = analyze("What would happen if the Fed raised rates?")
    assert a.evidence == "frame" and a.frame == "what_if"
    m = analyze("Compare the TCP protocol and the UDP protocol.").masked
    assert "compare" in m and "tcp" not in m.lower()


def test_content_words_are_lemmatised_topic_words():
    from bloombot.text.linguistics import content_words

    a = content_words("Describe the historical development of wind turbine in India.")
    b = content_words("Assess the viability of installing wind turbines in Tamil Nadu.")
    assert {"wind", "turbine"} <= a & b
    assert not {"describe", "assess"} & (a | b)  # Bloom verbs are not topic
