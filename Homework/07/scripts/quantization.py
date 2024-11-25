import torch
from typing import Tuple
from torch import Tensor, CharTensor


def absmax_quantization(x: Tensor) -> Tuple[float, CharTensor]:
    """
    Выполняет квантование тензора `x` с использованием метода максимального значения по модулю (AbsMax).

    Args:
        x (Tensor): Входной тензор для квантования.

    Returns:
        Tuple[float, CharTensor]: Кортеж, содержащий масштабный коэффициент `s` и квантованный тензор `x_q` типа `int8`.
            - `s` (float): Масштабный коэффициент, использованный для квантования.
            - `x_q` (CharTensor): Квантованный тензор с значениями типа `int8`.
    """
    s = 127.0 / torch.max(torch.abs(x)).item()
    x_q = torch.round(x * s).to(torch.int8)
    return s, x_q


def absmax_dequantization(s: float, x_q: Tensor) -> Tensor:
    """
    Выполняет деквантование тензора `x_q`, полученного методом AbsMax.

    Args:
        s (float): Масштабный коэффициент, использованный для квантования.
        x_q (Tensor): Квантованный тензор типа `int8`.

    Returns:
        Tensor: Восстановленный (деквантованный) тензор с типом `float`.
    """
    return x_q/s


def zeropoint_quantization(x: Tensor) -> Tuple[float, int, CharTensor]:
    """
    Выполняет квантование тензора `x` с использованием метода нулевой точки (Zero-Point Quantization).

    Args:
        x (Tensor): Входной тензор для квантования.

    Returns:
        Tuple[float, int, CharTensor]: Кортеж, содержащий масштабный коэффициент `s`, значение нулевой точки `z`,
            и квантованный тензор `x_q` типа `int8`.
            - `s` (float): Масштабный коэффициент, использованный для квантования.
            - `z` (int): Смещение (нулевая точка), использованное для квантования.
            - `x_q` (CharTensor): Квантованный тензор с значениями типа `int8`.
    """
    s = 255.0 / (torch.max(x) - torch.min(x)).item()
    z = -1 * s * torch.min(x).item() - 128.0
    x_q = torch.round(s * x + round(z)).to(torch.int8)
    return s, round(z), x_q


def zeropoint_dequantization(s: float, z: int, x_q: Tensor) -> Tensor:
    """
    Выполняет деквантование тензора `x_q`, полученного методом Zero-Point Quantization.

    Args:
        s (float): Масштабный коэффициент, использованный для квантования.
        z (int): Смещение (нулевая точка), использованное для квантования.
        x_q (Tensor): Квантованный тензор типа `int8`.

    Returns:
        Tensor: Восстановленный (деквантованный) тензор с типом `float`.
    """
    res = (x_q.float() - z)/s
    return res
