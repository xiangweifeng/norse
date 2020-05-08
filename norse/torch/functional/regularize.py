import torch
import numpy as np


def spike_count_lower(
    input_spikes : torch.Tensor,
    theta_l : float = 0.01,
    s_l : float = 1.0
):
    """Lower threshold spike count regularization as used
    in (https://arxiv.org/pdf/1910.07407.pdf).

    Parameters:
        input_spikes (torch.Tensor): Spike sequence to be regularized.
        theta_l (float): lower spike rate treshhold before regularization is applied.
        s_l (float): strength of the regularization.
    """
    seq_length = input_spikes.shape[0]
    N = np.sum(input_spikes.shape[1:])
    return s_l / N * torch.sum(torch.relu(1 / seq_length * torch.sum(input_spikes, axis=0) - theta_l)**2)


def spike_count_upper(
    input_spikes : torch.Tensor,
    theta_u : float = 100,
    s_u : float = 1.0
):
    """Upper threshold mean spike count regularization as used
    in (https://arxiv.org/pdf/1910.07407.pdf).

    Parameters:
        input_spikes (torch.Tensor): Spike sequence to be regularized.
        theta_u (float): lower spike rate treshhold before regularization is applied.
        s_u (float): strength of the regularization.
    """
    seq_length = input_spikes.shape[0]
    batch_size = input_spikes.shape[1]
    N = np.sum(input_spikes.shape[2:])
    return s_u / batch_size * torch.sum(torch.relu(1 / (seq_length * N) * torch.sum(input_spikes, axis=(0, *input_spikes.shape[2:])) - theta_u)**2)
