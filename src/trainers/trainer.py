import random
import time
from typing import Callable, List, Optional, Tuple

from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
import torch.nn as nn

from src.common.constants import *


class Trainer(nn.Module):

    def __init__(self,
                 problem_type: str,
                 weights: Optional[torch.tensor] = None,
                 checkpoint_path: str = None,
                 logger_func: Optional[Callable] = print,
                 test_on_train_dataset: bool = False):
        """ Trainer class

        :param problem_type: str, the problem to solve (REGRESSION or CLASSIFICATION)
        :param weights: torch.tensor, in the CLASSIFICATION case, the weight of each class. Optional
        """
        super(Trainer, self).__init__()
        self.problem_type = problem_type
        if problem_type == REGRESSION:
            self.criterion = self.custom_mse
            self.test_criterion = self.custom_mse
            self.str_test = 'MSE'
        elif problem_type == CLASSIFICATION:
            self.criterion = torch.nn.CrossEntropyLoss(weight=weights)
            self.test_criterion = self.custom_accuracy
            self.str_test = 'Accuracy'
        self.logger_func = logger_func
        self.test_on_train_dataset = test_on_train_dataset
        self.checkpoint_path = checkpoint_path

    def train_model(self,
                    brain: nn.Module,
                    optimizer,
                    train_data: List[torch.tensor],
                    test_data: List[torch.tensor],
                    epochs: int,
                    batch_size: int,
                    device: str,
                    extra_text: Optional[str]=None):
        """ Main flux of train/test

        :param brain: Model to train
        :param optimizer: Optimizer to use
        :param train_data: List of tensors to use for training
        :param test_data: List of tensors to use for testing
        :param epochs: int, number of epochs
        :param batch_size: int, batch size
        :param device: str, device to use
        :param extra_text: str
        :return:
        """
        start_train = time.time()

        # Preparamos el array de indices sobre el que nos apoyaremos para hacer shuffle en cada época
        tot = train_data[0].shape[0]
        tot_test = test_data[0].shape[0]
        epoch_shuffle_idx = [i for i in range(tot)]

        # For sobre las epocas
        best_res = float('inf') if self.problem_type == REGRESSION else -float('inf')
        best_res_str = ''
        epoch_metrics = []
        epoch_train_time = []
        epoch_test_time = []
        for epoch in range(epochs):
            new_extra_text = (extra_text + "EPOCH: {}/{} - ".format(epoch + 1, epochs) if extra_text is not None else
                              "EPOCH: {}/{} - ".format(epoch + 1, epochs))
            start_train_epoch_time = time.time()
            train_str = self.epoch_train(brain, optimizer, train_data, tot, epoch_shuffle_idx, batch_size, device, new_extra_text)
            epoch_train_time.append(time.time() - start_train_epoch_time)
            if self.test_on_train_dataset:
                _, new_extra_text = self.epoch_test(brain, train_data, tot, batch_size, device, TRAIN_TITLE, new_extra_text)
            start_test_epoch_time = time.time()
            epoch_res, test_str = self.epoch_test(brain, test_data, tot_test, batch_size, device, TEST_TITLE, new_extra_text)
            epoch_test_time.append(time.time() - start_test_epoch_time)
            epoch_metrics.append(epoch_res)
            if (((self.problem_type == REGRESSION) and (epoch_res.item() < best_res)) or
                ((self.problem_type == CLASSIFICATION) and (epoch_res.item() > best_res))):
                best_res = epoch_res.item()
                best_res_str = ' - BEST: {:.4f}'.format(best_res)
                if self.checkpoint_path is not None:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': brain.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_res
                    }, self.checkpoint_path)
            print(new_extra_text + train_str + test_str + best_res_str)

        if self.problem_type == REGRESSION:
            best_res, best_epoch = torch.tensor(epoch_metrics).min().item(), torch.argmin(torch.tensor(epoch_metrics)).item()
        else:
            best_res, best_epoch = torch.tensor(epoch_metrics).max().item(), torch.argmax(torch.tensor(epoch_metrics)).item()
        self.logger_func('Best result: {} - epoch: {} - time: {:.1f}'.format(best_res, best_epoch + 1, time.time() - start_train))
        return {METRIC: best_res,
                EPOCH: best_epoch + 1,
                TRAIN_TIME: torch.tensor(epoch_train_time).mean().item(),
                TEST_TIME: torch.tensor(epoch_test_time).mean().item()}

    def epoch_train(self,
                    brain: nn.Module,
                    optimizer,
                    data: List[torch.tensor],
                    dataset_size: int,
                    epoch_shuffle_idx: List[int],
                    batch_size: int,
                    device: str,
                    extra_text: Optional[str]=None) -> str:
        """ Epoch train

        :param brain: Model to train
        :param optimizer: Optimizer to use
        :param data: List of torch.tensor to use for training
        :param dataset_size: int, size of the training dataset
        :param epoch_shuffle_idx: list of indexes (used to shuffle the dataset in each epoch)
        :param batch_size: int
        :param device: str
        :param extra_text: str
        :return: str, the trace of training
        """
        start_epoch = time.time()

        # brain -> train mode
        brain.train()

        # Epoch shuffle
        random.shuffle(epoch_shuffle_idx)

        # For over the batches of the current epoch
        ini = 0
        fin = ini + batch_size
        train_loss_list = []
        title = TRAINING_TITLE if extra_text is None else extra_text + TRAINING_TITLE
        progress_bool = True # TODO: a parámetro!
        pbar = None
        if progress_bool:
            pbar = tqdm(total=int(dataset_size / batch_size) + 1, desc=title)

        while True:
            # Batch data preprocessing
            forward_input, targets = self.get_train_batch(data=data,
                                                          epoch_shuffle_idx=epoch_shuffle_idx,
                                                          ini=ini,
                                                          fin=fin,
                                                          device=device)
            # Optimization
            optimizer.zero_grad()
            predictions: torch.tensor = brain.forward(*forward_input)  # Shape: [batch_size, class_num]
            total_loss_function = self.criterion(predictions, targets.squeeze().detach())
            total_loss_function.backward()
            torch.nn.utils.clip_grad_value_(self.parameters(), 1.0)
            optimizer.step()

            # Trace
            train_loss_list.append(total_loss_function.detach().item())

            if progress_bool:
                pbar.update(1)
            ini = fin
            fin = ini + batch_size
            if fin > dataset_size:
                break

            del total_loss_function
            del predictions
            del targets
            del forward_input
            if device == 'cuda':
                torch.cuda.empty_cache()
        trace_txt = 'Train Loss: {:.10f} - Time: {:.4f} - '.format(torch.tensor(train_loss_list).mean(),
                                                                time.time() - start_epoch)
        del train_loss_list
        return trace_txt

    def epoch_test(self,
                   brain: nn.Module,
                   epoch_data_test: List[torch.tensor],
                   dataset_size: int,
                   batch_size: int,
                   device: str,
                   test_type: str,
                   extra_text: Optional[str]=None) -> Tuple[float, str]:
        """ Epoch test

        :param brain: nn.Module to test
        :param epoch_data_test: List of torch.tensor to use for testing
        :param dataset_size: int, size of the testing dataset
        :param batch_size: int
        :param device: str
        :param test_type: str to use in the logging
        :param extra_text: str
        :return: Tuple[float->metric, str->trace of test results]
        """
        start_test = time.time()

        # brain -> eval mode
        brain.eval()

        # For over the batches of the current epoch
        pre_list = []
        true_list = []
        title = test_type if extra_text is None else extra_text + test_type
        progress_bool = False  # TODO: a parámetro!
        pbar =None
        if progress_bool:
            pbar = tqdm(total=int(dataset_size / batch_size) + 1, desc=title)

        with torch.no_grad():
            ini = 0
            fin = ini + batch_size
            while True:
                # Batch data preprocessing
                forward_input, targets = self.get_test_batch(data=epoch_data_test,
                                                             ini=ini,
                                                             fin=fin,
                                                             device=device)

                # Test
                predictions: torch.tensor = brain.forward(*forward_input)
                pre_list = pre_list + predictions.detach().to('cpu').numpy().tolist()
                true_list = true_list + targets.detach().to('cpu').numpy().tolist()

                if progress_bool:
                    pbar.update(1)
                ini = fin
                fin = min(ini + batch_size, dataset_size)
                if ini == dataset_size:
                    break
                del predictions
                del targets
                del forward_input
                if device == 'cuda':
                    torch.cuda.empty_cache()
        test_metric = self.test_criterion(torch.tensor(pre_list), torch.tensor(true_list))
        trace_txt = '{} {}: {:.10f} - Time: {:.4f}'.format(test_type, self.str_test, test_metric,
                                                           time.time() - start_test)

        del pre_list, true_list
        return test_metric, trace_txt

    def get_train_batch(self, data, epoch_shuffle_idx, ini, fin, device):
        raise NotImplementedError

    def get_test_batch(self, data, ini, fin, device):
        raise NotImplementedError

    @staticmethod
    def custom_accuracy(predictions: torch.tensor, targets: torch.tensor) -> torch.tensor:
        """ Accuracy score

        :param predictions: torch.tensor, shape = [batch_size, class_num]
        :param targets: torch.tensor, shape = [batch_size, class_num]
        :return: float
        """
        preds = torch.argmax(predictions, dim=1).cpu()
        trues = targets.squeeze().cpu()
        return torch.tensor(accuracy_score(trues, preds))

    @staticmethod
    def custom_mse(predictions: torch.tensor, targets: torch.tensor) -> torch.tensor:
        loss = torch.nn.MSELoss()
        return loss(predictions.squeeze(), targets.squeeze())          
