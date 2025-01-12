

from abc import ABC, abstractmethod
from jaxtyping import Int
import torch


class BasePairMatching(ABC):
    @abstractmethod
    def __call__(self, seq_len: int) -> torch.Tensor:
        """
        Args:
            seq_len: sequence length used for the pair matching
        Returns:
            torch.Tensor: Pairs of indices of shape (N, 2).
        """
        pass
    
class NeighboursPairMatching(ABC):
    @abstractmethod
    def __call__(self, seq_len: int) -> torch.Tensor:
        """
        Args:
            seq_len: sequence length used for the pair matching
        Returns:
            torch.Tensor: Pairs of indices of shape (N, 2).
        """
        if seq_len%2 != 0:
            raise ValueError("Neighbours pair matching requires even sequence length")
        return torch.Tensor([[i, i+1] for i in range(seq_len-1)])
    
class FirstCenteredPairMatching(ABC):
    @abstractmethod
    def __call__(self, seq_len: int) -> torch.Tensor:
        """
        Args:
            seq_len: sequence length used for the pair matching
        Returns:
            torch.Tensor: Pairs of indices of shape (N, 2).
        """
        return torch.Tensor([[0, i] for i in range(1, seq_len)])