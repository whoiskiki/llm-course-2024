import torch
from torch import no_grad
from .compute_reward import compute_reward


def generate_with_reward_guidance(
    main_model,
    main_tokenizer,
    reward_model,
    reward_tokenizer,
    prompt="",
    N=16,
    max_length=50,
    device="cpu",
):
    """
    Generate text samples using a main model and select the best sample based on a reward model's guidance.

    Parameters:
    - main_model: The language model used to generate text samples.
    - main_tokenizer: The tokenizer for main_model.
    - reward_model: The model used to compute reward scores for the generated samples.
    - reward_tokenizer: The tokenizer for reward_model.
    - prompt (str, optional): The input text to guide the generation. Default is an empty string.
    - N (int, optional): The number of text samples to generate. Default is 16.
    - max_length (int, optional): The maximum length of the generated text. Default is 50.
    - device (str, optional): The device on which the computation should be performed. Default is 'cpu'.

    Returns:
    str: The generated text sample with the highest reward score.
    """

    # Ensure models are on the correct device
    #main_model.to(device)
    #reward_model.to(device)

    # Tokenize the input prompt for the main model
    main_inputs = main_tokenizer(prompt, return_tensors="pt").to(device)

    # Generate N text samples
    outputs = main_model.generate(
        **main_inputs,
        max_length=max_length,
        num_return_sequences=N,
        do_sample=True,  # Use sampling for diversity
        temperature=0.7,  # Adjust temperature for creativity
    )

    # Decode the generated outputs
    generated_texts = [main_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    # Evaluate reward scores for each sample
    reward_scores = []
    with no_grad():
        for text in generated_texts:
            # Tokenize the text for the reward model
            reward_inputs = reward_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
            # Compute reward score
            score = compute_reward(reward_model, reward_inputs)
            reward_scores.append(score)

    # Select the text with the highest reward score
    best_index = torch.argmax(torch.tensor(reward_scores))
    best_text = generated_texts[best_index]

    return best_text
