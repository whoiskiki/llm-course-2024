import random
from fsm import FSM, build_odd_zeros_fsm


def get_valid_tokens(vocab: dict[int, str], eos_token_id: int, fsm: FSM, state: int) -> list[int]:
    """Filter tokens from the vocabulary based on the given state in the FSM.  
    1. Retain only tokens that can be achieved from the given state.  
    2. If the current state is terminal, then add the EOS token.

    Args:
        vocab (dict): vocabulary, id to token
        eos_token_id (int): index of EOS token
        fsm (FSM): Finite-State Machine
        state (int): start state
    Returns:
        valid tokens (list): list of possible tokens
    """
    valid_tokens = []

    for token_id, token in vocab.items():
        if token_id == eos_token_id:
            continue

        next_state = fsm.move(token, state)
        if next_state is not None:
            valid_tokens.append(token_id)

    if fsm.is_terminal(state):
        valid_tokens.append(eos_token_id)

    return valid_tokens


def random_generation() -> str:
    """Structured generation based on Odd-Zeros FSM with random sampling from possible tokens.

    Args:
    Returns:
        generation (str): A binary string with an odd number of zeros.
    """
    # Define our vocabulary
    vocab = {0: "[EOS]", 1: "0", 2: "1"}
    eos_token_id = 0
    # Init Finite-State Machine
    fsm, state = build_odd_zeros_fsm()

    # List with generate tokens
    tokens: list[int] = []
    # Sample until EOS token
    while True:
        # 1. Get valid tokens
        valid_tokens = get_valid_tokens(vocab, eos_token_id, fsm, state)
        # 2. Get next token
        next_token = random.choice(valid_tokens)
        tokens.append(next_token)

        # 3. End generation or move to next iteration
        if next_token == eos_token_id:
            break
        state = fsm.move(vocab[next_token], state)

    return "".join([vocab[it] for it in tokens if it != eos_token_id])


if __name__ == "__main__":
    print(random_generation())
