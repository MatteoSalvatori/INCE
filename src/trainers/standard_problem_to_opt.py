import torch.optim as optim

from src.models.brain import Brain
from src.models.brain_struct import *
from src.trainers.trainer_factory import TrainerFactory


class StandardProblemToOpt(object):

    def __init__(self, dataset_data, brain_params, problem_type: str, problem_size: int,
                 target_embedding: bool):
        self.dataset_data = dataset_data
        self.brain_params = brain_params
        self.problem_type = problem_type
        self.problem_size = problem_size
        self.target_embedding = target_embedding
        self.weights = None
        if problem_type == CLASSIFICATION:
            self.weights = dataset_data.weights

    def __call__(self, *args, **kwargs):
        # Model
        print('Building brain...')
        (general_info, common_info, input_graph_embedding, _,
         encoder_gnn_info, _, decoder_info) = build_brain_structs(self.dataset_data, self.brain_params)
        # Brain
        brain = Brain(general_info=general_info,
                      common_info=common_info,
                      input_graph_embedding=input_graph_embedding,
                      encoder_gnn_info=encoder_gnn_info,
                      decoder_info=decoder_info,
                      device=self.brain_params[TRAINER_PARAMS][DEVICE]).to(self.brain_params[TRAINER_PARAMS][DEVICE])
        print(brain)
        print("Number of trainable parameters: {}"
              .format(sum(p.numel() for p in brain.parameters() if p.requires_grad)))
        optimizer = optim.Adam(brain.parameters(), self.brain_params[TRAINER_PARAMS][TRAIN_LEARNING_RATE])
        
        # Trainer
        print('Building trainer...')
        trainer_type = (TrainerFactory.TRAINER_WITH_TARGET_EMBEDDING if self.target_embedding else
                        TrainerFactory.TRAINER_WITHOUT_TARGET_EMBEDDING)
        weights = self.weights if self.weights is None else self.weights.to(self.brain_params[TRAINER_PARAMS][DEVICE])
        checkpoint_path = self.brain_params[TRAINER_PARAMS][PATH]
        trainer = TrainerFactory.get_trainer(trainer_type=trainer_type,
                                             problem_type=self.problem_type,
                                             weights=weights,
                                             checkpoint_path=checkpoint_path)
        # Train-Test
        print('Start training...')
        return trainer.train_model(brain=brain,
                                   optimizer=optimizer,
                                   train_data=[self.dataset_data.x_train, self.dataset_data.y_train],
                                   test_data=[self.dataset_data.x_test, self.dataset_data.y_test],
                                   epochs=self.brain_params[TRAINER_PARAMS][EPOCHS],
                                   batch_size=self.brain_params[TRAINER_PARAMS][BATCH_SIZE],
                                   device=self.brain_params[TRAINER_PARAMS][DEVICE])
