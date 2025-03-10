from beartype import beartype
from jaxtyping import jaxtyped
from networkx import complement
from jaxtyping import Float
import torch
import torch.nn as nn


class F1LossWithLogits(nn.Module):

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        target: Float[torch.Tensor, "B N 1 X Y Z"],
        output: Float[torch.Tensor, "B N 1 X Y Z"],
    ) -> Float[torch.Tensor, "B N"]:
        """
        B: Batch Size
        N: Number of pairs
        X, Y, Z: Spatial dimensions


        # Loosely based on
        # https://www.kaggle.com/code/rejpalcz/best-loss-function-for-f1-score-metric#Optimal-loss-function---macro-F1-score
        """
        # Convert target and output to binary values if needed
        # Assuming output is already passed through sigmoid or similar
        # to produce values between 0 and 1

        output = torch.sigmoid(output)

        # Flatten the spatial dimensions (X, Y, Z) for computation
        target_flat = target.view(target.shape[0], target.shape[1], 1, -1)
        output_flat = output.view(output.shape[0], output.shape[1], 1, -1)

        # Calculate true positives, true negatives, false positives, false negatives
        tp = torch.sum(target_flat * output_flat, dim=3)
        tn = torch.sum((1 - target_flat) * (1 - output_flat), dim=3)
        fp = torch.sum((1 - target_flat) * output_flat, dim=3)
        fn = torch.sum(target_flat * (1 - output_flat), dim=3)

        # Calculate precision and recall
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall + self.epsilon)

        # Handle NaN values
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)

        # Remove the singleton dimension (third dimension)
        f1 = f1.squeeze(2)

        # Return 1 - F1 score (since we want to minimize loss)
        return 1 - f1
