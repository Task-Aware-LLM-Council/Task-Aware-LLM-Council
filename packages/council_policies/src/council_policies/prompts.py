"""
Prompt templates for council voting and evaluation policies.
"""

# ---------------------------------------------------------------------------
# Voter Prompt
# ---------------------------------------------------------------------------
# Used when each model in the council is asked to vote on the best answer.
# Answers are labelled A/B/C (not by model name) to prevent self-preference bias.
# ---------------------------------------------------------------------------

VOTER_PROMPT_TEMPLATE = """\
You are a strict, unbiased evaluator in a council of AI models.

A question was answered by three different sources. The answers are labelled A, B, and C — \
you do not know which model produced which answer.

Your job is to read all three answers carefully and vote for the BEST one.

======================
QUESTION:
{question}
======================

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

ANSWER C:
{answer_c}

======================
EVALUATION CRITERIA:
======================

Check each answer for:
- Correctness (especially for math, logic, and code)
- Completeness — does it fully address the question?
- Clarity — is it easy to understand?
- Logical soundness — are the reasoning steps valid?

Watch for:
- Factual errors or hallucinations
- Missing steps or incomplete reasoning
- Incorrect assumptions
- Code that would not run correctly

======================
DOMAIN-SPECIFIC RULES:
======================

- MATH: Prioritize correct final answer and valid steps.
- CODING: Prioritize correctness, runability, and efficiency.
- GENERAL KNOWLEDGE: Prioritize factual accuracy and completeness.
- WRITING / ENGLISH: Prioritize clarity, coherence, and usefulness.

======================
VOTING RULES:
======================

- You MUST vote for exactly ONE answer: A, B, or C.
- Do NOT factor in which answer you think you wrote — evaluate purely on quality.
- If multiple answers are equally good, pick the most complete and precise one.
- If all answers are wrong or poor, vote for the least incorrect one.

======================
OUTPUT FORMAT (STRICT — follow exactly):
======================

Vote: <A / B / C>

Confidence: <High / Medium / Low>

Reason:
<Brief comparison of all three answers. Explain why the winner is better than the others. \
Be specific about errors or gaps in the losing answers.>
"""


# ---------------------------------------------------------------------------
# Final Aggregator Prompt (optional: used after votes are collected)
# ---------------------------------------------------------------------------
# If votes are tied or you want a meta-judge to produce a final answer,
# this prompt can be used with a single arbitrator call.
# ---------------------------------------------------------------------------

AGGREGATOR_PROMPT_TEMPLATE = """\
You are the final arbitrator in a council of AI models.

A question was answered by three models (labelled A, B, C). \
Each model then voted for the best answer. Here are the votes:

======================
QUESTION:
{question}
======================

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

ANSWER C:
{answer_c}

======================
VOTES CAST:
======================

{votes_summary}

======================
YOUR TASK:
======================

1. Review the votes and the answers.
2. Select the final winner, taking the votes into account but correcting for any clear errors.
3. Produce a final, polished answer — fix any mistakes from all models if needed.

======================
OUTPUT FORMAT (STRICT):
======================

Winner: <A / B / C / NONE>

Confidence: <High / Medium / Low>

Reason:
<Explain the final decision. Note if you overrode the vote majority and why.>

Final Answer:
<The best possible answer to the question. Correct and complete.>
"""


def build_voter_prompt(question: str, answer_a: str, answer_b: str, answer_c: str) -> str:
    """Build the voter prompt with answers anonymized as A, B, C."""
    return VOTER_PROMPT_TEMPLATE.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        answer_c=answer_c,
    )


def build_aggregator_prompt(
    question: str,
    answer_a: str,
    answer_b: str,
    answer_c: str,
    votes_summary: str,
) -> str:
    """Build the aggregator prompt after all votes have been collected.

    votes_summary should be a human-readable string, e.g.:
        'Model 1 voted: A\\nModel 2 voted: A\\nModel 3 voted: B'
    """
    return AGGREGATOR_PROMPT_TEMPLATE.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        answer_c=answer_c,
        votes_summary=votes_summary,
    )
