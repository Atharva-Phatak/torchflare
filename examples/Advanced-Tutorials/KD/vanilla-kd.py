"""Vanilla Knowledge distillation using Torchflare.
This example only shows how to modify the training script for KD.
"""
from typing import Dict

import torch
import torch.nn.functional as F

from torchflare.experiments import Experiment


class KDExperiment(Experiment):
    def __init__(self, temperature, alpha, **kwargs):
        super(KDExperiment, self).__init__(**kwargs)
        self.temperature = temperature
        self.alpha = alpha

    def get_grad_params(self, config):
        grad_params = list(self.state.model["student_network"].parameters())
        return grad_params

    def train_step(self) -> Dict:
        self.backend.zero_grad(optimizer=self.state.optimizer)
        x, y = self.batch[self.input_key], self.batch[self.target_key]

        with torch.no_grad():
            teacher_outputs = self.state.model["teacher_network"](x)

        # setting self.preds to student_network outputs since we need to monitor train accuracy of student network.
        student_outputs = self.state.model["student_network"](x)

        student_loss = self.state.criterion["student_criterion"](student_outputs, y)
        student_loss = (1 - self.alpha) * student_loss

        distillation_loss = self.state.criterion["distillation_criterion"](
            F.softmax(student_outputs / self.temperature, dim=1),
            F.softmax(teacher_outputs / self.temperature, dim=1),
        )

        distillation_loss = (self.alpha * self.temperature * self.temperature) * distillation_loss
        loss = student_loss + distillation_loss

        self.backend.backward_loss(loss=loss)
        self.backend.optimizer_step(optimizer=self.state.optimizer)
        return {self.prediction_key: student_outputs, self.loss_key: loss}

    def val_step(self) -> Dict:
        x, y = self.batch[self.input_key], self.batch[self.target_key]
        student_outputs = self.state.model["student_network"](x)
        student_loss = self.state.criterion["student_criterion"](student_outputs, y)
        return {self.prediction_key: student_outputs, self.loss_key: student_loss}
