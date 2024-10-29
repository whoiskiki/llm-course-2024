import torch
import torch.nn.functional as F

def scaled_dot_product_gqa(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool = True, need_weights: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product attention in grouped manner.

    Args:
        query (torch.Tensor): Query tensor of shape [batch size; seq len; num heads; hidden dim]
        key (torch.Tensor): Key tensor of shape [batch size; kv seq len; num kv heads; hidden dim]
        value (torch.Tensor): Value tensor of shape [batch size; kv seq len; num kv heads; hidden dim]
        is_causal (bool): Whether causal mask of attention should be used
        need_weights (bool): Whether attention weights should be returned

    Returns:
        2-tuple of torch.Tensor:
            - Attention output with shape [batch size; seq len; num heads; hidden dim]
            - (Optional) Attention weights with shape [batch size; num heads; seq len; kv seq len].
                Only returned if 'need_weights' is True.
    """
    batch_size, seq_len, num_heads, hidden_dim = query.shape
    kv_seq_len = key.size(1)
    num_kv_heads = key.size(2)

    if num_heads != num_kv_heads:
        if num_heads > num_kv_heads:
            factor = num_heads // num_kv_heads
            key = key.repeat_interleave(factor, dim=2)
            value = value.repeat_interleave(factor, dim=2)
        else:
            raise ValueError("Количество heads у queries должно быть <= heads у keys")

    attention_scores = torch.einsum('bqhd,bkhd->bhqk', query, key)/(hidden_dim**0.5)

    if is_causal:
        causal_mask = torch.tril(torch.ones((seq_len, kv_seq_len), device=query.device)).bool()
        attention_scores = attention_scores.masked_fill(~causal_mask, float('-inf'))

    attention_weights = F.softmax(attention_scores, dim=-1)
    attention_output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, value)

    if need_weights:
        return attention_output, attention_weights
    return attention_output, None
