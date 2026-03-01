"""Knowledge distillation for continual learning (optional CMKL enhancement).

If EWC + replay isn't sufficient, knowledge distillation preserves the old
model's "dark knowledge" about entity relationships. The old model (teacher)
generates soft targets on new task data, and the new model (student) learns
from both hard targets and soft targets.

Usage:
    from src.continual.distillation import KnowledgeDistillation
    kd = KnowledgeDistillation(teacher_model, student_model, temperature=2.0)
    distill_loss = kd.compute_distillation_loss(batch)
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class KnowledgeDistillation:
    """Knowledge distillation from old model to new model.

    Args:
        teacher: Old model (frozen) that provides soft targets.
        student: New model being trained.
        temperature: Softmax temperature for distillation.
        alpha: Weight balancing hard vs. soft targets.
    """

    def __init__(
        self,
        teacher: "nn.Module",
        student: "nn.Module",
        temperature: float = 2.0,
        alpha: float = 0.5,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha

    def compute_distillation_loss(self, batch: dict) -> "torch.Tensor":
        """Compute KD loss between teacher and student on a batch.

        L_total = alpha * L_hard + (1 - alpha) * T^2 * L_soft

        Args:
            batch: Training batch dict.

        Returns:
            Combined distillation loss tensor.
        """
        raise NotImplementedError("Phase 4 (optional): Implement distillation loss")
