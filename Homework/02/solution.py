import torch
import torch.nn.functional as F


def compute_attention(queries, keys, values) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    keys- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    values- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    """
    E = queries.size()[-1]
    scores = torch.matmul(queries,  keys.mT)/(E**0.5)
    att_scores = F.softmax(scores, dim=-1)
    output = torch.matmul(att_scores, values)
    return output


def compute_multihead_attention(queries, keys, values, projection_matrix) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    keys- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    values- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    projection_matrix- (N_HEADS*DIM_PER_HEAD, N_HEADS*DIM_PER_HEAD)
    """
    batch_size, n, seq_length, dim_per_head = queries.size()
    print(n, dim_per_head)
    scores = torch.matmul(queries, keys.mT)/torch.sqrt(torch.tensor(dim_per_head, dtype=torch.float32))

    scores = scores - scores.max(dim=-1, keepdim=True)[0]  # стабилизируем экспоненту
    weights = F.softmax(scores, dim=-1) #Берем последнюю ось
    output = torch.matmul(weights, values)

    concat = output.transpose(1, 2).reshape(batch_size, seq_length, n*dim_per_head)
    print(f'size: {projection_matrix.size()}')
    result = torch.matmul(concat, projection_matrix.T)

    return result

def compute_rotary_embeddings(x) -> torch.Tensor:
    """
    X - (BATCH_SIZE, SEQ_LENGTH, N_HEADS, DIM_PER_HEAD)
    Сначала генерируем theta, потом вычисляем m*theta
    """
    batch_size, seq_length, n, dim_per_head = x.size()

    theta = 1. / (10000**(torch.arange(0, dim_per_head, 2).float()/dim_per_head))
    seq_idx = torch.arange(seq_length, device=x.device).float()

    m_theta = torch.einsum('i,j->ij', seq_idx, theta)

    th_cos = m_theta.cos()
    th_sin = m_theta.sin()

    x_even = x[..., ::2]
    x_odd = x[..., 1::2]

    th_cos = th_cos.unsqueeze(0).unsqueeze(2)
    th_sin = th_sin.unsqueeze(0).unsqueeze(2)

    rotated_even = x_even * th_cos - x_odd * th_sin
    rotated_odd = x_even * th_sin + x_odd * th_cos

    pos_emm = torch.stack([rotated_even, rotated_odd], dim=-1).reshape(batch_size, seq_length, n, dim_per_head)

    return pos_emm
