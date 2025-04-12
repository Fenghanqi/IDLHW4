import torch

def PadMask(padded_input, input_lengths):
    """ 
    Create a mask to identify non-padding positions. 
    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
        input_lengths: The actual lengths of each sequence before padding, shape (N,).
    Returns:
        A boolean mask tensor with shape (N, T), where: 
            - padding positions are marked with True 
            - non-padding positions are marked with False.
    """
    # Get batch size and sequence length
    batch_size = padded_input.size(0)
    seq_length = padded_input.size(1)
    
    # Create a tensor of indices [0, 1, 2, ..., seq_length-1]
    indices = torch.arange(seq_length, device=padded_input.device).unsqueeze(0).expand(batch_size, -1)
    
    # Create mask: position indices >= length are padding (True)
    # shape: (batch_size, seq_length)
    mask = indices >= input_lengths.unsqueeze(1)
    
    return mask

def CausalMask(padded_input):
    """ 
    Create a mask to identify non-causal positions. 
    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
    
    Returns:
        A boolean mask tensor with shape (T, T), where: 
            - non-causal positions (don't attend to) are marked with True 
            - causal positions (can attend to) are marked with False.
    """
    # Get sequence length
    seq_length = padded_input.size(1)
    
    # Create a lower triangular matrix (including the diagonal)
    # where lower triangular (including diagonal) is 0, and upper triangular is 1
    mask = torch.triu(torch.ones(seq_length, seq_length, device=padded_input.device), diagonal=1).bool()
    
    return mask

