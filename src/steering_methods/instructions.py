INSTR_TEMPLATE = """
Instructions:\n
Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity. With this in mind, please continue the following text.\n\n
Text:
[prompt]
"""

INSTR_FORMAL = """
Instructions:\n
Assist with silliness and informality. Respond with utmost utility yet securely. Avoid formal content. Ensure replies are colloquial with a casual writing style. With this in mind, please continue the following text.\n\n
Text:
[prompt]
"""

INSTR_FORMAL_CONTINUOUS = """
Instructions:\n
Text can have different levels of formality. Newspapers and encyclopedias have high formality ratings and text messages may have low formality ratings. 
The maximum formality rating is 3, and the minimum formality rating is -3. A rating of 0 is neutral.\n
Example of -3: ``(LOL) jus kidding... the answer to your question is GAS PRICES!!!".\n
Example of 3: ``With the everpresent elderly community in South Broward, we are better poised to serve that community by having a "community representative" on our board."
With this in mind, please continue the following text so that it has a formality rating [S].
Text: 
[prompt]
"""

def transform_prompt(prompt: str, experiment: str, continuous_tune: bool, S=None):
    if not continuous_tune or S is None:
        return INSTR_TEMPLATE.replace('[prompt]', prompt) if experiment in ('toxicity', 'negativity') else INSTR_FORMAL.replace('[prompt]', prompt)

    if continuous_tune and S is not None:
        return INSTR_FORMAL_CONTINUOUS.replace('[prompt]', prompt).replace('[S]', str(S))

def transform_dataset(data, experiment, continuous_tune, S):
    """Returns a list of instruction-fitted prompts.

    Args:
        data (list(str)): list of natural language prompts
    """
    return [transform_prompt(prompt, experiment, continuous_tune, S) for prompt in data]