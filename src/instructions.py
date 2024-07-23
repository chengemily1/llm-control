INSTR_TEMPLATE = """
Instructions:\n
Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity. With this in mind, please continue the following text.\n\n
Text:
[prompt]
"""

def transform_prompt(prompt: str):
    return INSTR_TEMPLATE.replace('[prompt]', prompt)

def transform_dataset(data):
    """Returns a list of instruction-fitted prompts.

    Args:
        data (list(str)): list of natural language prompts
    """
    return [transform_prompt(prompt) for prompt in data]