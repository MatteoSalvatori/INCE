from typing import Optional

import torch

from src.trainers.trainer import Trainer
from src.trainers.trainer_te import TrainerTE
from src.trainers.trainer_wo_te import TrainerWOTE


class TrainerFactory(object):
    TRAINER_WITH_TARGET_EMBEDDING = 'trainer_with_target_embedding'
    TRAINER_WITHOUT_TARGET_EMBEDDING = 'trainer_without_target_embedding'

    @staticmethod
    def get_trainer(trainer_type: str,
                    problem_type: str,
                    weights: Optional[torch.tensor] = None,
                    checkpoint_path: Optional[str] = None) -> Trainer:
        if trainer_type == TrainerFactory.TRAINER_WITH_TARGET_EMBEDDING:
            trainer = TrainerTE(problem_type, weights, checkpoint_path)
        elif trainer_type == TrainerFactory.TRAINER_WITHOUT_TARGET_EMBEDDING:
            trainer = TrainerWOTE(problem_type, weights, checkpoint_path)
        else:
            print('{} trainer is not implemented!'.format(trainer_type))
            trainer = None
        return trainer
