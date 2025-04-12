import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable
from ..data import H4Tokenizer

'''
TODO: Implement the `generate_greedy` and optionally the `generate_beam` methods of the `SequenceGenerator` class.

This file implements text generation strategies for transformer language models:

1. Greedy Search: Always selects the most likely next token
   - Simple but can lead to repetitive or suboptimal outputs
   - Useful for deterministic generation

2. Beam Search: Maintains top-k most likely sequences at each step
   - Explores multiple possible sequences in parallel
   - Often produces higher quality outputs than greedy search
   - More computationally intensive

3. Sampling with Filtering: Uses probabilistic sampling with constraints
   - Temperature: Controls randomness of sampling
   - Top-k: Limits sampling to k most likely tokens
   - Top-p (nucleus): Samples from minimal set of tokens comprising p probability mass
   - Useful for creative and diverse generation

Implementation Notes:
1. Helper Methods:
   - _apply_repeat_penalty: Penalizes repeated tokens
   - _filter_logits: Applies temperature and filtering
   - post_process_sequence: Handles EOS token truncation

2. Generation Methods:
   - generate_greedy: Implements basic greedy decoding
   - generate_beam: Implements beam search
   - generate_sample: Implements filtered sampling

3. Each generation method should:
   - Handle proper input validation
   - Track sequence scores
   - Handle EOS token detection
   - Support early stopping
'''

class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies.
    Supports greedy search, beam search, and sampling with top-k/nucleus filtering.
    """
    def __init__(
            self,
            score_fn: Callable,
            tokenizer: H4Tokenizer,
            max_length: int,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the sequence generator.
        
        Args:
            score_fn: Function that returns logits for next token prediction
            tokenizer: Tokenizer instance for handling token conversions
            max_length: Maximum sequence length to generate
            device: Device to run generation on
        """
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def _apply_repeat_penalty(
            self,
            logits: torch.Tensor,
            sequences: torch.Tensor,
            penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits based on tokens in sequences.
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size) or (batch_size, beam_width, vocab_size)
            sequences: Sequences tensor of shape (batch_size, sequence_length) or (batch_size, beam_width, sequence_length)
            penalty: Repetition penalty value
        Returns:
            Logits tensor with repetition penalty applied
        """
        if penalty == 1.0:
            return logits
        
        # Handle both regular and beam search shapes
        if logits.dim() == 2:
            # Greedy search: (batch_size, vocab_size)
            for idx in range(sequences.size(0)):
                unique_tokens = torch.unique(sequences[idx])
                logits[idx, unique_tokens] = logits[idx, unique_tokens] / torch.where(
                    logits[idx, unique_tokens] > 0,
                    torch.full_like(logits[idx, unique_tokens], penalty),
                    torch.full_like(logits[idx, unique_tokens], 1.0/penalty)
                )
        else:
            # Beam search: (batch_size, beam_width, vocab_size)
            for batch_idx in range(sequences.size(0)):
                for beam_idx in range(sequences.size(1)):
                    unique_tokens = torch.unique(sequences[batch_idx, beam_idx])
                    logits[batch_idx, beam_idx, unique_tokens] = logits[batch_idx, beam_idx, unique_tokens] / torch.where(
                        logits[batch_idx, beam_idx, unique_tokens] > 0,
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], penalty),
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], 1.0/penalty)
                    )
        
        return logits

    def _filter_logits(
            self,
            logits: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> torch.Tensor:
        """Apply temperature, top-k, and top-p filtering to logits."""
        logits = logits / temperature
        if top_k > 0:
            top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            indices_to_remove = logits < top_k_logits[..., -1:]
            logits[indices_to_remove] = float('-inf')
        if top_p < 1.0:
            log_probs = torch.log_softmax(logits, dim=-1)
            sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True)
            cumulative_probs = torch.cumsum(torch.exp(sorted_log_probs), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        return logits

    def generate_greedy(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using greedy search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        
        # Initialize scores and finished flag
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        
        # Generate tokens until max_length or all sequences are finished
        for _ in range(self.max_length - x.size(1)):
            # Check if all sequences have finished
            if finished.all():
                break
                
            # Get logits from the model
            logits = self.score_fn(x)  # (batch_size, vocab_size)
            
            # Apply repeat penalty to penalize repeated tokens
            logits = self._apply_repeat_penalty(logits, x, repeat_penalty)
            
            # Apply temperature scaling
            logits = logits / temperature
            
            # Get log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Get the most likely next token
            next_token_scores, next_tokens = torch.max(log_probs, dim=-1)  # (batch_size,)
            
            # Update scores for unfinished sequences
            scores = torch.where(finished, scores, scores + next_token_scores)
            
            # Append next tokens to the sequences
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)  # (batch_size, seq_len + 1)
            
            # Update finished flag when EOS token is generated
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos
        
        return x, scores

    def generate_beam(
            self,
            x: torch.Tensor,
            beam_width: int,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using beam search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            beam_width: Number of beams to use
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, beam_width, sequence_length) where each sequence in a beam set is sorted by score
             - scores is of shape (batch_size, beam_width)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        
        # Special case for beam_width=1 (equivalent to greedy search)
        if beam_width == 1:
            sequences, scores = self.generate_greedy(x, temperature, repeat_penalty)
            return sequences.unsqueeze(1), scores.unsqueeze(1)
        
        # Initialize variables
        batch_size = x.size(0)
        vocab_size = self.tokenizer.vocab_size
        
        # Initialize scores with zeros
        scores = torch.zeros(batch_size, beam_width, device=x.device)
        
        # Initialize finished flags
        finished = torch.zeros(batch_size, beam_width, dtype=torch.bool, device=x.device)
        
        # Initialize beam sequences with the input sequences
        # First, get logits for initial tokens
        logits = self.score_fn(x)  # (batch_size, vocab_size)
        
        # Apply repeat penalty
        logits = self._apply_repeat_penalty(logits, x, repeat_penalty)
        
        # Apply temperature
        logits = logits / temperature
        
        # Get log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)  # (batch_size, vocab_size)
        
        # Get top beam_width tokens and their scores
        topk_scores, topk_tokens = torch.topk(log_probs, beam_width, dim=-1)  # (batch_size, beam_width)
        
        # Update scores with initial token scores
        scores = topk_scores
        
        # Expand input sequences for beams
        x = x.unsqueeze(1).expand(batch_size, beam_width, x.size(-1))  # (batch_size, beam_width, seq_len)
        
        # Append top-k tokens to create initial beam sequences
        x = torch.cat([
            x, 
            topk_tokens.unsqueeze(-1)  # (batch_size, beam_width, 1)
        ], dim=-1)  # (batch_size, beam_width, seq_len + 1)
        
        # Update finished flag for sequences that generated EOS token
        finished = finished | (topk_tokens == self.tokenizer.eos_id)
        
        # Continue generating until max_length or all beams are finished
        for t in range(1, self.max_length - x.size(-1)):
            # Check if all beams for all batches are finished
            if finished.all():
                break
                
            # Prepare storage for next token logits for all beams
            next_token_scores = []
            
            # Process each beam
            for beam_idx in range(beam_width):
                # Skip if all sequences in this beam are finished
                if finished[:, beam_idx].all():
                    # Fill with negative infinity to ensure these won't be selected
                    next_token_scores.append(
                        torch.full((batch_size, vocab_size), float('-inf'), device=x.device)
                    )
                    continue
                
                # Get logits for next tokens for this beam
                beam_logits = self.score_fn(x[:, beam_idx])  # (batch_size, vocab_size)
                
                # Apply repeat penalty
                beam_logits = self._apply_repeat_penalty(beam_logits, x[:, beam_idx], repeat_penalty)
                
                # Apply temperature
                beam_logits = beam_logits / temperature
                
                # Get log probabilities
                beam_log_probs = torch.log_softmax(beam_logits, dim=-1)  # (batch_size, vocab_size)
                
                # Update for finished sequences: set all logits to -inf except for EOS token
                for batch_idx in range(batch_size):
                    if finished[batch_idx, beam_idx]:
                        beam_log_probs[batch_idx, :] = float('-inf')
                        beam_log_probs[batch_idx, self.tokenizer.eos_id] = 0.0  # No additional score for EOS
                
                next_token_scores.append(beam_log_probs)
            
            # Stack all beam scores: (batch_size, beam_width, vocab_size)
            next_token_scores = torch.stack(next_token_scores, dim=1)
            
            # Calculate cumulative scores for all possible next beams
            # shape: (batch_size, beam_width, vocab_size)
            cum_scores = scores.unsqueeze(-1) + next_token_scores
            
            # Reshape for easier top-k selection
            # shape: (batch_size, beam_width * vocab_size)
            cum_scores_flat = cum_scores.view(batch_size, -1)
            
            # Select top-k scores and their indices
            topk_cum_scores, topk_indices = torch.topk(cum_scores_flat, beam_width, dim=-1)
            
            # Convert flat indices to beam and token indices
            beam_indices = topk_indices // vocab_size  # Which beam did this come from
            token_indices = topk_indices % vocab_size  # Which token was selected
            
            # Update scores
            scores = topk_cum_scores
            
            # Create new beam sequences
            new_x = []
            for batch_idx in range(batch_size):
                batch_new_x = []
                for new_beam_idx in range(beam_width):
                    # Get originating beam
                    orig_beam_idx = beam_indices[batch_idx, new_beam_idx].item()
                    # Get token to append
                    token = token_indices[batch_idx, new_beam_idx].item()
                    
                    # Get the sequence from the originating beam
                    seq = x[batch_idx, orig_beam_idx].clone()
                    
                    # Append new token if this isn't a finished sequence
                    if not finished[batch_idx, orig_beam_idx]:
                        seq = torch.cat([seq, torch.tensor([token], device=seq.device)])
                    
                    batch_new_x.append(seq)
                
                # Pad sequences to the same length
                max_len = max(seq.size(0) for seq in batch_new_x)
                padded_batch_new_x = []
                for seq in batch_new_x:
                    if seq.size(0) < max_len:
                        # Pad with EOS
                        pad_size = max_len - seq.size(0)
                        padding = torch.full((pad_size,), self.tokenizer.eos_id, device=seq.device)
                        seq = torch.cat([seq, padding])
                    padded_batch_new_x.append(seq)
                
                new_x.append(torch.stack(padded_batch_new_x))
            
            # Stack all new sequences
            x = torch.stack(new_x)  # (batch_size, beam_width, new_seq_len)
            
            # Update finished flag
            new_finished = torch.zeros_like(finished)
            for batch_idx in range(batch_size):
                for new_beam_idx in range(beam_width):
                    # Get originating beam
                    orig_beam_idx = beam_indices[batch_idx, new_beam_idx].item()
                    # Check if this was already finished
                    was_finished = finished[batch_idx, orig_beam_idx]
                    # Check if the new token is EOS
                    token = token_indices[batch_idx, new_beam_idx].item()
                    is_eos = (token == self.tokenizer.eos_id)
                    # Update finished flag
                    new_finished[batch_idx, new_beam_idx] = was_finished | is_eos
            
            finished = new_finished
        
        # Sort beams by score
        sorted_scores, sorted_indices = torch.sort(scores, dim=1, descending=True)
        
        # Rearrange sequences according to sorted scores
        sorted_x = torch.zeros_like(x)
        for batch_idx in range(batch_size):
            sorted_x[batch_idx] = x[batch_idx, sorted_indices[batch_idx]]
        
        return sorted_x, sorted_scores

    def generate_sample(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using sampling with top-k and nucleus filtering.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            top_k: Number of top-k tokens to sample from
            top_p: Proportion of top-p tokens to sample from
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be > 0 and <= 1.0")
        
        # Initialize scores and finished flag
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        for _ in range(self.max_length - x.size(1)):
            # Check if all sequences have finished
            if finished.all():
                break
            # Get logits and apply filtering
            next_scores = self.score_fn(x) # (batch_size, vocab_size)
            filtered_logits = self._filter_logits(next_scores, temperature, top_k, top_p)
            log_probs = torch.log_softmax(filtered_logits, dim=-1)
            
            # We need probabilities for multinomial sampling
            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1) # (batch_size,)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1) # (batch_size,)
            # Update scores only for unfinished sequences
            scores = torch.where(finished, scores, scores + token_scores)
            # Append next tokens
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1) # (batch_size, seq_len + 1)
            # Check if any sequence has reached EOS 
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos
        return x, scores

    @staticmethod
    def post_process_sequence(seq: torch.Tensor, tokenizer: H4Tokenizer) -> torch.Tensor:
        """
        Post process sequences to remove content after EOS token.
        Args:
            seq: Input tensor of shape (batch_size, sequence_length) or (sequence_length)
            tokenizer: Tokenizer instance for handling token conversions
        Returns:
            if seq is a single sequence, return a tensor of same shape with sequence truncated at EOS
            if seq is a batch of sequences, return a list of tensors with each sequence truncated at first EOS
        """
        # Handle single sequence case
        if seq.dim() == 1:
            eos_indices = (seq == tokenizer.eos_id).nonzero()
            if len(eos_indices) > 0:
                end_idx = eos_indices[0].item() + 1
                return seq[:end_idx]
            return seq
        
        # Handle batched sequences
        eos_mask = seq == tokenizer.eos_id  # (batch_size, sequence_length)
        # Find first EOS token in each sequence
        eos_indices = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        # Create sequence mask that includes everything up to and including first EOS
        seq_mask = eos_indices.cumsum(dim=1).eq(0) | eos_indices
        # Apply mask and pack sequences
        return [s[:m.sum()] for s, m in zip(seq, seq_mask)]