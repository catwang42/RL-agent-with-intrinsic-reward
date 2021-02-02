import numpy
import torch
import torch.nn.functional as F

from torch_ac.algos.base import BaseAlgo

class A2CAlgoIntrinsic(BaseAlgo):
    """The Advantage Actor-Critic algorithm."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.01, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-8, preprocess_obss=None, reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)

    def update_parameters(self, exps):
        # Compute starting indexes

        inds = self._get_starting_indexes()

        # Get update values

        update_entropy, update_value, update_policy_loss, update_value_loss, update_loss = self.get_update_values(exps, inds)

        # Intrinsic Reward save existing state

        old_parameters = {}
        for name, param in self.acmodel.named_parameters():
            old_parameters[name] = param.detach().numpy().copy()

        # Update update values

        update_entropy /= self.recurrence
        update_value /= self.recurrence
        update_policy_loss /= self.recurrence
        update_value_loss /= self.recurrence
        update_loss /= self.recurrence

        # Update actor-critic

        update_loss, update_grad_norm = self.update_parameters_helper(update_loss)

        # second round of update values

        update_entropy, update_value, update_policy_loss, update_value_loss, update_loss_2 = self.get_update_values(exps, inds)

        # Final update for intrinsic reward

        new_parameters = {}
        for name, param in self.acmodel.named_parameters():
            new_parameters[name] = param.detach().numpy().copy()
        norm_diff  = []
        for index in range(len(old_parameters.keys())):
            if index == 0 or index == 2 or index == 4: # weight layers of the CNN only
                key = list(old_parameters.keys())[index]
                norm_diff.append(numpy.linalg.norm(new_parameters[key] - old_parameters[key]))

        update_loss_2 += (numpy.mean(norm_diff))

        # Update update values

        update_loss_2 /= self.recurrence + 1

        # Update actor-critic

        _, _ = self.update_parameters_helper(update_loss_2)

        # Log some values

        logs = {
            "entropy": update_entropy,
            "value": update_value,
            "policy_loss": update_policy_loss,
            "value_loss": update_value_loss,
            "grad_norm": update_grad_norm
        }

        return logs

    def update_parameters_helper(self, update_loss):

        # Update actor-critic

        self.optimizer.zero_grad()
        update_loss.backward()
        update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return update_loss, update_grad_norm

    def get_update_values(self, exps, inds):
        # Initialize update values

        update_entropy = 0
        update_value = 0
        update_policy_loss = 0
        update_value_loss = 0
        update_loss = 0

        # Initialize memory

        if self.acmodel.recurrent:
            memory = exps.memory[inds]

        for i in range(self.recurrence):
            # Create a sub-batch of experience

            sb = exps[inds + i]

            # Compute loss

            if self.acmodel.recurrent:
                dist, value, memory = self.acmodel(sb.obs, memory * sb.mask)
            else:
                dist, value = self.acmodel(sb.obs)

            entropy = dist.entropy().mean()

            policy_loss = -(dist.log_prob(sb.action) * sb.advantage).mean()

            value_loss = (value - sb.returnn).pow(2).mean()

            loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

            # Update batch values

            update_entropy += entropy.item()
            update_value += value.mean().item()
            update_policy_loss += policy_loss.item()
            update_value_loss += value_loss.item()
            update_loss += loss

        return update_entropy, update_value, update_policy_loss, update_value_loss, update_loss


    def _get_starting_indexes(self):
        """Gives the indexes of the observations given to the model and the
        experiences used to compute the loss at first.
        The indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`. If the model is not recurrent, they are all the
        integers from 0 to `self.num_frames`.
        Returns
        -------
        starting_indexes : list of int
            the indexes of the experiences to be used at first
        """

        starting_indexes = numpy.arange(0, self.num_frames, self.recurrence)
        return starting_indexes