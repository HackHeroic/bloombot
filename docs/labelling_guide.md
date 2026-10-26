# Labelling guide (for teachers)

Use this when you (a) label your institution's questions to create a profile, or (b) check
the rewrites in `results/tables/rewrite.csv` (`human_a_*`, `human_b_*` columns). Two people
should label independently, then compare; report Cohen's kappa (quadratic-weighted for levels).

## Ask one question: what must the student *do* to answer well?

| Level | The student must … | Typical tasks |
|---|---|---|
| Remember | retrieve facts, terms, definitions, lists from memory | define, list, state, name, "what is" |
| Understand | show meaning in their own words: explain, summarise, classify, compare at a descriptive level, give an example | explain, describe, summarise, illustrate |
| Apply | carry out a known procedure on a given case | calculate, solve, use X to find Y, trace, draw a given diagram from data |
| Analyze | break material into parts and work out how they relate; find causes, structure, assumptions | analyse the given data, differentiate, examine why |
| Evaluate | judge against criteria and defend the judgement | justify, critique, assess, which is better and why |
| Create | produce something new and coherent | design, propose, develop, formulate |

## Rules

1. **The main instruction decides, not any verb in the sentence.** "Explain how engineers
   *evaluate* a bridge" is Understand.
2. **Several tasks → the highest one really demanded.** "Define a stack and *design* an
   algorithm …" is Create.
3. **Familiarity matters.** If the "explanation" was given in class word for word, students
   only remember it; label it as your institution intends and be consistent. Write the rule
   down (e.g. "Explain X from the syllabus = Remember") — that is exactly the convention a
   profile learns.
4. **Maths "evaluate"** (an integral, an expression) is Apply.
5. **No Bloom verb?** Ask the same question: "What would happen if …" (Analyze/Understand),
   "Do you agree …" (Evaluate), "Who/when/where …" (Remember).
6. Ignore marks and difficulty: a hard calculation is still Apply.
7. Mark a question `unsure` rather than guessing; unsure items are excluded from training.

## Checking a rewrite

For each accepted rewrite record: the level you would give it (`human_*_level`) and whether
it is a usable exam question on the same topic (`human_*_ok`: yes/no). A rewrite counts as a
success only if both reviewers agree it is at the target level and usable.
