from pathlib import Path

import numpy as np
import pytest

from bloombot.eval.metrics import scores
from bloombot.models.linear import BloomLex, LexiconRule
from bloombot.models.llm import distribution_from_logprobs
from bloombot.pipeline import blueprint as bp
from bloombot.pipeline.intake import parse_text
from bloombot.pipeline.rewrite import Rewriter
from bloombot.pipeline.schemas import Blueprint, LevelBound, Paper, Question, RewriteJob

ROOT = Path(__file__).resolve().parents[1]


def test_paper_parse():
    p = parse_text((ROOT / "data" / "papers" / "sample_os_paper.txt").read_text())
    ids = [q.id for q in p.questions]
    assert ids == ["1a", "1b", "1c", "1d", "1e", "2", "3", "4a", "4b", "5", "6"]
    q = {q.id: q for q in p.questions}
    assert q["1a"].marks == 2 and q["1a"].course_outcome == "CO1" and q["1a"].tagged_level == "Remember"
    assert q["1a"].text == "Define a process."
    assert q["2"].group == q["3"].group is not None          # OR alternatives
    assert q["4a"].context and q["4a"].context.startswith("Consider")
    assert p.total_marks == 50                                 # OR pair counted once


@pytest.mark.parametrize("model", [LexiconRule(), BloomLex(), BloomLex(ordinal=True)])
def test_models_return_distributions(model, toy):
    q, y = toy
    model.fit(q, y)
    p = model.predict_proba(["Define recursion.", "Design a new compiler."])
    assert p.shape == (2, 6)
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
    assert list(model.predict(["Define recursion.", "Design a new compiler."])) == [1, 6]


def test_metrics():
    s = scores([1, 2, 3, 4], [1, 2, 3, 4])
    assert s["accuracy"] == 1 and s["qwk"] == pytest.approx(1)
    s = scores([1, 2, 3, 4], [2, 3, 4, 5])
    assert s["accuracy"] == 0 and s["within_one"] == 1 and s["mae"] == 1


def test_logprob_distribution():
    lp = [{"token": '{"', "logprob": 0}, {"token": "level", "logprob": 0}, {"token": '":', "logprob": 0},
          {"token": "4", "logprob": -0.2, "top_logprobs": [{"token": "4", "logprob": -0.2}, {"token": "2", "logprob": -1.8}]}]
    p = distribution_from_logprobs(lp, 4)
    assert p.argmax() == 3 and 0.7 < p[3] < 0.9 and p[1] > 0
    assert distribution_from_logprobs(None, 2).argmax() == 1
    # non-ASCII digits among the alternatives must be ignored, not crash
    lp[-1]["top_logprobs"] += [{"token": "\u2082", "logprob": -3.0}, {"token": "\uff12", "logprob": -3.0}]
    assert distribution_from_logprobs(lp, 4).argmax() == 3


def _paper():
    return Paper(questions=[
        Question(id="1", text="Define a stack.", marks=10), Question(id="2", text="List the types of memory.", marks=10),
        Question(id="3", text="State Ohm's law.", marks=10), Question(id="4", text="Explain paging.", marks=10, group="g"),
        Question(id="5", text="Describe the water cycle.", marks=10, group="g"),
    ])


def test_report_and_plan(assessor):
    paper = _paper()
    preds = assessor.assess([q.text for q in paper.questions])
    blue = Blueprint(id="t", name="t", bounds=[LevelBound(levels=["Remember"], max_share=0.40)])
    rep = bp.report(paper, preds, blue)
    # 30 Remember marks out of 40 effective (the OR pair shares one 10-mark slot)
    assert rep.distribution["Remember"] == pytest.approx(0.75)
    assert rep.compliant is False
    jobs = bp.plan(rep, blue)
    assert len(jobs) == 2 and all(j.from_level == "Remember" for j in jobs)


def test_rewrite_template_offline(assessor, settings):
    rw = Rewriter(assessor, None, settings)
    res = rw.run(RewriteJob(question_id="1", original="Define a binary search tree.", from_level="Remember",
                            to_level="Create"))
    assert res.attempts and res.attempts[0].composer == "template"
    assert "binary search tree" in res.attempts[0].question
    assert {c.name for c in res.attempts[0].checks} >= {"level", "form", "topic", "not_duplicate"}


def test_table_layout_paper_with_whole_question_or():
    p = parse_text((ROOT / "data" / "papers" / "sample_table_paper.txt").read_text())
    assert [q.id for q in p.questions] == ["1a", "1b", "1c", "2a", "2b", "3a", "3b", "4a", "4b"]
    q = {q.id: q for q in p.questions}
    assert q["1a"].text == "Define a compiler. List the phases of a compiler."
    assert (q["2b"].marks, q["2b"].tagged_level, q["4b"].tagged_level) == (14, "Apply", "Evaluate")
    assert q["1a"].group == q["2b"].group and q["1a"].option != q["2b"].option
    assert p.total_marks == 40
    assert sum(bp.question_weights(p)) == pytest.approx(40)


# --- regressions from the independent review -------------------------------------------------------
@pytest.mark.parametrize("text,total,groups", [
    ("11(a) Explain paging. [16]\nOR\n11(b) Explain segmentation. [16]\n12(a) Define a process. [16]\n", 32, 1),
    ("11(a) Define X. [6]\n11(b) Explain Y. [10]\nOR\n12(a) Define Z. [6]\n12(b) Explain W. [10]\n13. Compare A and B. [4]\n", 20, 1),
])
def test_inline_numbering_or_groups(text, total, groups):
    p = parse_text(text)
    assert p.total_marks == total
    assert len({q.group for q in p.questions if q.group}) == groups
    assert sum(bp.question_weights(p)) == pytest.approx(total)
    assert p.questions[-1].group is None or "13" not in p.questions[-1].id  # compulsory question stays outside


def test_printed_tags_do_not_leak_into_text():
    q = parse_text("1. Explain TCP congestion control in detail. CO2 L2 [10]\n").questions[0]
    assert q.text == "Explain TCP congestion control in detail."
    assert (q.marks, q.tagged_level, q.course_outcome) == (10, "Understand", "CO2")


def test_wrapped_line_starting_with_number_is_continuation():
    p = parse_text("1. A vessel contains\n2 moles of an ideal gas at 300 K. Find the pressure. [5]\n2. Define entropy. [5]\n")
    assert [q.id for q in p.questions] == ["1", "2"] and "2 moles" in p.questions[0].text


def test_logprobs_ignore_digits_inside_reasoning():
    toks = ['{"', "reasoning", '":', ' "', "a", " level", " ", "4", " task", '",', ' "', "level", '":', " ", "2", "}"]
    lp = [{"token": t, "logprob": 0.0} for t in toks]
    lp[7]["top_logprobs"] = [{"token": "4", "logprob": -0.01}]
    lp[14]["top_logprobs"] = [{"token": "2", "logprob": -0.1}, {"token": "3", "logprob": -2.5}]
    p = distribution_from_logprobs(lp, 2)
    assert p.argmax() == 1 and p[2] > 0


def test_stratified_sample_keeps_level_shares():
    import pandas as pd
    from bloombot.eval.runner import _stratified

    df = pd.DataFrame({"source": ["s"] * 1000, "level": [2] * 600 + [1] * 300 + [6] * 100, "uid": range(1000)})
    s = _stratified(df, 100)
    assert len(s) == 100 and s.level.value_counts().to_dict() == {2: 60, 1: 30, 6: 10}


def test_profiles_do_not_share_llm_retriever(assessor, settings, toy):
    from bloombot.models.llm import LLMClassifier
    from bloombot.pipeline.assess import Assessor

    q, y = toy
    base = Assessor(assessor.lexis, assessor.bank, settings.model_copy(update={"encoder": "hashing"}),
                    llm=LLMClassifier("none", shots=2))
    before = base.llm.retriever
    prof = base.with_examples(["Explain paging."] * 5, [1] * 5, "inst")
    assert base.llm is not prof.llm and base.llm.retriever is before
    assert len(prof.llm.retriever.questions) == len(q) + 5 and len(before.questions) == len(q)


def test_svm_calibration_with_rare_class(toy):
    from bloombot.models.linear import TfidfSVM

    q, y = toy
    q2, y2 = list(q) + ["Invent a new game."], np.append(y, 6)
    keep = [i for i, v in enumerate(y2) if v != 6][:24] + [len(y2) - 1, len(y2) - 2]  # 2 Create examples
    m = TfidfSVM().fit([q2[i] for i in keep], y2[keep])
    assert m.fallback_ is False


def test_pool_rules():
    from bloombot.pipeline.assess import pool

    lex = np.array([0.05, 0.6, 0.3, 0.03, 0.01, 0.01])  # Lexis: Understand, Apply second
    llm = np.array([0.0, 0.0, 0.02, 0.0, 0.98, 0.0])   # an overconfident LLM: Evaluate
    lin = pool([lex, llm], [1, 0.5], "linear")
    assert np.isclose(lin.sum(), 1) and lin.argmax() == 1  # the mixture keeps Lexis' answer
    geo = pool([lex, llm], [1, 0.5], "geometric")
    assert geo.argmax() == 2  # log-linear pooling settles on a level neither voter chose
    assert np.allclose(pool([lex, llm], [1, 0], "linear"), lex)  # weight 0 = evidence only


def test_explicit_q_headings_ignore_bare_numbers_and_guessed_marks():
    # A LaTeX worksheet: matrix rows ("1.0", "0") start lines, subscripts trail sentences, no marks.
    text = ("Q1: Sample Problem 1\nA decoder has query s = [\n1.0\n0.5\n] and states h.\n"
            "(a) Compute the alignment scores.\n(b) Compute precision, recall and F for each class.\n1\n"
            "Q2: Sample Problem 2\n0 A second setting.\n(a) Prove that the softmax is invariant to shifts.\n")
    p = parse_text(text)
    assert [q.id for q in p.questions] == ["1a", "1b", "2a"]
    assert all(q.marks is None for q in p.questions)


def test_llm_failure_on_one_question_abstains(monkeypatch):
    from bloombot.models.llm import LLMClassifier, LLMError

    llm = LLMClassifier("none")

    def classify_one(q):
        if "fail" in q:
            raise LLMError("Server disconnected without sending a response.")
        return {"question": q, "level": 2, "proba": [0, 1, 0, 0, 0, 0], "reasoning": None}

    monkeypatch.setattr(llm, "classify_one", classify_one)
    outs = llm.classify_many(["Explain paging.", "this one will fail"])
    assert outs[0]["level"] == 2
    assert outs[1]["level"] is None and "disconnected" in outs[1]["error"]


def test_cancelled_assessment_stops_llm_calls(assessor, settings, monkeypatch):
    import threading

    from bloombot.models.llm import LLMClassifier
    from bloombot.pipeline.assess import Assessor, Cancelled

    llm = LLMClassifier("none")
    calls = []
    monkeypatch.setattr(llm, "classify_one", lambda q: calls.append(q) or {"question": q, "level": 2,
                                                                          "proba": [0, 1, 0, 0, 0, 0]})
    a = Assessor(assessor.lexis, assessor.bank, settings.model_copy(update={"llm_trigger_confidence": 1.01}), llm)
    stop = threading.Event()
    stop.set()
    with pytest.raises(Cancelled):
        a.assess(["Explain paging.", "Define a stack."], use_llm=True, cancel=stop)
    assert calls == []
    assert len(a.assess(["Explain paging."], use_llm=True)[0].votes) == 3  # not cancelled: Socrates votes


@pytest.mark.parametrize("header,minutes,marks", [
    ("Time: 3 Hrs                 Max. Marks: 100", 180, 100),
    ("Duration: 90 minutes   Maximum Marks : 50", 90, 50),
    ("Time Allowed: 1½ hours  Total marks: 60", 90, 60),
    ("(Time: 1 hour 30 minutes)", 90, None),
    ("Course: Operating Systems", None, None),
])
def test_header_duration_and_max_marks(header, minutes, marks):
    p = parse_text(header + "\nQ1. Define a stack. [2]\nQ2. Explain paging. [5]\n")
    assert p.duration_minutes == minutes and p.max_marks == marks
    assert len(p.questions) == 2


def _report(assessor, settings, paper, pace="typical"):
    from bloombot.pipeline.blueprint import load_blueprints
    from bloombot.pipeline.conductor import Conductor

    return Conductor(assessor, settings).assess_paper(paper, load_blueprints()["balanced-ug"], use_llm=False, pace=pace)


def test_time_estimate_actions_and_or_groups(assessor, settings):
    from bloombot.pipeline.feasibility import actions
    from bloombot.pipeline.schemas import Paper, Question

    assert [a.split(":")[1] for a in actions("Prove that x > 0, and verify it numerically.", "Analyze")] == ["prove", "compute"]
    assert actions("Construct the DFA for (a|b)*abb.", "Apply") == ["construct:compute"]
    qs = [Question(id="1", text="Define a stack.", marks=2),
          Question(id="2", text="Explain paging with a diagram.", marks=10, group="g", option="2"),
          Question(id="3", text="Define a page.", marks=10, group="g", option="3")]
    rep = _report(assessor, settings, Paper(questions=qs, duration_minutes=60))
    t = {x.id: x for x in rep.timing.questions}
    assert t["2"].high > 15  # a 10-mark written answer: planning and ~250 words
    # only the longer OR-alternative counts
    assert rep.timing.total_high == pytest.approx(t["1"].high + t["2"].high, abs=0.2)
    slow = _report(assessor, settings, Paper(questions=qs, duration_minutes=60), pace="slower")
    assert slow.timing.total_high > rep.timing.total_high


def test_too_long_paper_and_bad_marking_are_not_ready(assessor, settings):
    from bloombot.pipeline.schemas import Paper, Question

    qs = [Question(id="1", text="Define a stack.", marks=10),
          Question(id="2", text="Prove that the problem is undecidable and derive the bound.", marks=1),
          Question(id="3", text="Explain paging.", marks=5), Question(id="4", text="Explain segmentation.", marks=5),
          Question(id="5", text="Compute the average waiting time.", marks=5)]
    rep = _report(assessor, settings, Paper(questions=qs, duration_minutes=10, max_marks=30))
    assert rep.timing.verdict == "too_long" and rep.ready is False
    checks = {c.name: c for c in rep.marking}
    assert checks["Total marks"].ok is False and "26" in checks["Total marks"].detail
    assert checks["Marks match the work"].ok is False
    assert set(checks["Marks match the work"].questions) == {"1", "2"}
    assert "OR alternatives are equivalent" not in checks  # no OR-groups in this paper


def test_paper_without_marks_or_duration(assessor, settings):
    from bloombot.pipeline.schemas import Paper, Question

    rep = _report(assessor, settings, Paper(questions=[Question(id="1", text="Explain paging.")]))
    assert rep.timing.verdict == "unknown"
    assert rep.marking[0].ok is None and len(rep.marking) == 1


def test_dataset_explorer_hides_non_public_sources(tmp_path):
    import pandas as pd

    from bloombot.data.explore import _load, bundle, query

    rows = [("yahya2012:1", "yahya2012", "Define a stack.", "Remember"),
            ("yahya2012:2", "yahya2012", "Design a database.", "Create"),
            ("indian_boards:1", "indian_boards", "Find the first term.", "Apply")]
    pd.DataFrame(rows, columns=["uid", "source", "question", "label"]).to_parquet(tmp_path / "questions.parquet")
    _load.cache_clear()
    page = query(tmp_path)
    assert page.total == 2 and {i.source for i in page.items} == {"yahya2012"}
    boards = next(s for s in page.sources if s.id == "indian_boards")
    assert boards.public is False and boards.rows == 1 and boards.per_level["Apply"] == 1
    assert query(tmp_path, source="indian_boards").items == []
    assert query(tmp_path, q="DATABASE").total == 1 and query(tmp_path, level="Remember").total == 1
    assert all(i["source"] != "indian_boards" for i in bundle(tmp_path)["items"])
    _load.cache_clear()


def test_quiz_headings_marks_options_and_footer():
    text = (
        "NLP Quiz\n25 questions · Answer all questions\n"
        "QUESTION 1/25 · 2 marks\nA student claims that skip-gram learns rare words better.\n"
        "Which correction is the most accurate?\nA It does, because each occurrence\nmakes several updates\n"
        "B It does not\nC Only with negative sampling\nD Only for frequent words\n"
        "NLP Quiz Page 1\n"
        "QUESTION 2/25 · 4 marks\nCompute the memory needed for the score matrices.\nA 32 MiB\nB 256 MiB\nC 512 MiB\n"
    )
    paper = parse_text(text)
    q1, q2 = paper.questions
    assert (q1.marks, q2.marks) == (2, 4) and paper.total_marks == 6
    assert q1.text.startswith("A student claims") and "/25" not in q1.text and "Page" not in q1.text
    assert q1.choices == ["It does, because each occurrence makes several updates", "It does not",
                          "Only with negative sampling", "Only for frequent words"]
    assert q2.text == "Compute the memory needed for the score matrices." and len(q2.choices) == 3


def test_multiple_choice_has_no_writing_time():
    from bloombot.pipeline.feasibility import question_time

    open_q = question_time("1", "Explain the attention mechanism.", "Understand", 0, 1.0, marks=10)
    mcq = question_time("2", "Explain the attention mechanism.", "Understand", 0, 1.0, marks=10, choices=["a b", "c d", "e f"])
    assert mcq.high < open_q.high and any("choose 1 of 3" in a for a in mcq.actions)


def test_claude_model_names_use_the_anthropic_client(monkeypatch):
    pytest.importorskip("anthropic")
    from bloombot.models.llm import AnthropicClient, OllamaClient, _strict, classification_schema, llm_client

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert isinstance(llm_client("claude-sonnet-5-5"), AnthropicClient)
    assert isinstance(llm_client("qwen3:4b-instruct-2507-q4_K_M"), OllamaClient)
    schema = _strict(classification_schema(cot=True))
    assert schema["additionalProperties"] is False and schema["properties"]["level"]["enum"] == [1, 2, 3, 4, 5, 6]
