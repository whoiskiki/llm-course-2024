def create_prompt(sample: dict) -> str:
    """
    Generates a prompt for a multiple choice question based on the given sample.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.

    Returns:
        str: A formatted string prompt for the multiple choice question.
    """
    sub = sample['subject']
    q = sample['question']
    choices = sample['choices']

    lit = ['A', 'B', 'C', 'D']

    str1 = f"The following are multiple choice questions (with answers) about {sub}.\n{q}\n"
    a = '\n'.join(map('. '.join, zip(lit, choices)))
    b = f"\nAnswer:"

    return str1 + a + b


def create_prompt_with_examples(sample: dict, examples: list, add_full_example: bool = False) -> str:
    """
    Generates a 5-shot prompt for a multiple choice question based on the given sample and examples.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.
        examples (list): A list of 5 example dictionaries from the dev set.
        add_full_example (bool): whether to add the full text of an answer option

    Returns:
        str: A formatted string prompt for the multiple choice question with 5 examples.
    """
    lit = ['A', 'B', 'C', 'D']
    result = ''

    for i in range(len(examples)):
        sub = examples[i]['subject']
        q = examples[i]['question']
        choices = examples[i]['choices']
        ans = examples[i]['answer']

        str1 = f"The following are multiple choice questions (with answers) about {sub}.\n{q}\n"
        a = '\n'.join(map('. '.join, zip(lit, choices)))

        if add_full_example:
            b = f"\nAnswer: {lit[ans]}. {choices[ans]}\n\n"
        else:
            b = f"\nAnswer: {lit[ans]}\n\n"

        result += str1 + a + b

    res = create_prompt(sample)

    return result + res
