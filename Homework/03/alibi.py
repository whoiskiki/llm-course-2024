import torch
import math

def compute_alibi(num_heads: int, seq_len: int) -> torch.Tensor:
    """
    Compute ALiBi for a sequence.

    ALiBi can be used not only with causal models.
    In this case, the biases will be symmetrical about the diagonal up to the sign.

    Args:
        num_heads (int): Number of attention heads.
        seq_len (int): Sequence length.

    Returns:
        torch.Tensor: A tensor containing ALiBi to be added to attention scores.
    """
    def get_slopes(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    if math.log2(num_heads).is_integer():
        slopes = get_slopes(num_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(
            math.log2(num_heads))
        slopes = get_slopes(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][:num_heads - closest_power_of_2]

    alibi = torch.zeros((num_heads, seq_len, seq_len))

    for head in range(num_heads):
        for i in range(seq_len):
            for j in range(seq_len):
                alibi[head, i, j] = slopes[head] * (j - i)

    return alibi


if __name__ == "__main__":
    bias = compute_alibi(4, 4)
    print(bias)
