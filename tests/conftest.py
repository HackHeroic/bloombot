import numpy as np
import pytest

from bloombot.config import Settings
from bloombot.models.linear import BloomLex
from bloombot.pipeline.assess import Assessor, Bank

# A tiny labelled set, enough for the models to learn verb -> level.
TOY = [
    ("Define a stack.", 1), ("List the types of memory.", 1), ("State Ohm's law.", 1), ("What is a compiler?", 1),
    ("Name the planets of the solar system.", 1),
    ("Explain how paging works.", 2), ("Summarize the causes of inflation.", 2), ("Describe the water cycle.", 2),
    ("Explain why the sky is blue.", 2), ("Classify the given animals into vertebrates and invertebrates.", 2),
    ("Calculate the current in the circuit.", 3), ("Solve the quadratic equation.", 3),
    ("Use Dijkstra's algorithm to find the shortest path.", 3), ("Apply the formula to the given data.", 3),
    ("Compute the mean of the sample.", 3),
    ("Compare TCP and UDP.", 4), ("Differentiate between a process and a thread.", 4),
    ("Analyze the causes of the 2008 financial crisis.", 4), ("Examine the structure of the given poem.", 4),
    ("Distinguish between weather and climate.", 4),
    ("Justify the use of nuclear power.", 5), ("Evaluate the effectiveness of the policy.", 5),
    ("Critique the design of this experiment.", 5), ("Assess whether the argument is valid.", 5),
    ("Judge which algorithm is better for small inputs.", 5),
    ("Design a database for a library.", 6), ("Propose a new method to reduce traffic.", 6),
    ("Develop a marketing plan for a startup.", 6), ("Create a lesson plan on fractions.", 6),
    ("Formulate a hypothesis about plant growth.", 6),
]


@pytest.fixture(scope="session")
def toy():
    q, y = zip(*TOY)
    return list(q), np.array(y)


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    return Settings(artifacts=tmp_path_factory.mktemp("artifacts"), lexis_model="bloomlex", encoder="hashing",
                    llm="", llm_policy="never", neighbours=3)


@pytest.fixture(scope="session")
def assessor(toy, settings):
    q, y = toy
    lexis = BloomLex(ordinal=True, raw_text=True).fit(q, y)
    bank = Bank(list(q), y, ["toy"] * len(q), np.zeros(len(q), bool))
    return Assessor(lexis, bank, settings, llm=None)
