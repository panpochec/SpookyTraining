import torch
from spookynet import SpookyNet
import math
import random
import json
import time
import datetime
import numpy as np


class SpookyNetBatch:

    def __init__(self, device='cpu'):
        """
        Initializing class with internal parameters for class set as (mimics SpookyNet):
        N: numbers of molecules in a batch
        Z: atomic numbers of atoms
        R: positions of atoms
        E: energies of molecules
        F: gradients for atoms, in loss functions they are reversed to obtain forces
        Q: charges of molecules
        U: dipole moments of molecules
        S: spins for molecules (not using for now)
        batch_seg: utility for assigning Z and R to molecules

        """
        self.N = 0
        self.Z = []
        self.R = []
        self.E = []
        self.Q = []
        self.S = []
        self.F = []
        self.U = []
        self.names = []
        self.batch_seg = []
        self.idx_i = []
        self.idx_j = []
        self.loss_helper = []
        self.device = torch.device(device)

    def toTensor(self):
        """
        Translates internal parameters to tensors
        :return:
        """
        self.S = torch.zeros(self.N, dtype=torch.float32,
                             device=self.device)  # not using spins for now, hence zeroes
        self.Z = torch.tensor(self.Z, dtype=torch.int64, device=self.device)
        self.R = torch.tensor(self.R, dtype=torch.float32, device=self.device, requires_grad=True)
        self.F = torch.tensor(self.F, dtype=torch.float32, device=self.device)
        self.E = torch.tensor(self.E, dtype=torch.float32, device=self.device)
        self.Q = torch.tensor(self.Q, dtype=torch.float32, device=self.device)
        self.U = torch.tensor(self.U, dtype=torch.float32, device=self.device)
        self.idx_i = torch.tensor(self.idx_i, dtype=torch.int64, device=self.device)
        self.idx_j = torch.tensor(self.idx_j, dtype=torch.int64, device=self.device)
        self.batch_seg = torch.tensor(self.batch_seg, dtype=torch.int64, device=self.device)
        self.loss_helper = torch.tensor(self.loss_helper, dtype=torch.float32, device=self.device)
        return self


def get_idx(R):
    """
    From SpookyNet, check in source
    """
    N = len(R)
    idx = torch.arange(N, dtype=torch.int64)
    idx_i = idx.view(-1, 1).expand(-1, N).reshape(-1)
    idx_j = idx.view(1, -1).expand(N, -1).reshape(-1)
    # exclude self-interactions
    nidx_i = idx_i[idx_i != idx_j]
    nidx_j = idx_j[idx_i != idx_j]
    return nidx_i.numpy(), nidx_j.numpy()


def logging_data(model_tolog, optimizer_tolog, scheduler_to_log, rmse_to_log, training_set_to_log,
                 validation_set_to_log, save_to_name='checkpoint.pth', n_batch=100, mean_tolog=0, std_devtolog=0):
    """
    Function used to log data from training

    :param model_tolog: model to log
    :param optimizer_tolog: optimizer to log
    :param scheduler_to_log: scheduler to log
    :param rmse_to_log: rmse of a logged point
    :param training_set_to_log: training set, used to extract the size of the sets
    :param validation_set_to_log:  validation set, used to extract the size of the sets
    :param save_to_name: name used for checkpoint file, checkpoint.pth by default
    :param n_batch: size of a single batch
    :return:
    """
    # Date logging
    datetime_object = datetime.datetime.now()
    year = str(datetime_object.year)
    month = str(datetime_object.month)
    day = str(datetime_object.day)
    hour = str(datetime_object.hour)
    minute = str(datetime_object.minute)
    if int(minute) < 10:
        minute = '0' + str(datetime_object.minute)
    logging_date = hour + ':' + minute + ', ' + day + '-' + month + '-' + year

    # Log text, used to provide info on the model stored in checkpoint file
    log_text = []
    log_text.append('Mean for training ' + str(mean_tolog) + ', standard deviation for training' + str(std_devtolog) + '\n')
    log_text.append('Log data for ' + save_to_name + ', recorded on ' + logging_date + '\n')
    log_text.append('Optimizer: ' + str(type(optimizer_tolog).__name__) + ', epoch ' + str(scheduler_to_log.last_epoch)
                    + '\n')
    log_text.append('Current RMSE [eV]: ' + str(rmse_to_log) + ', best point: ' + str(scheduler_to_log.best) + '\n')
    log_text.append('Current learning rate: ' + str(optimizer_tolog.param_groups[0]['lr']) + '\n')
    log_text.append('Training set size: ' + str(len(training_set_to_log) * n_batch) + '\n')
    log_text.append('Validation set size: ' + str(len(validation_set_to_log) * n_batch) + '\n')

    logepoch_text = []
    logepoch_text.append('Epoch ' + str(scheduler_to_log.last_epoch) + ' recorded on ' + logging_date + '\n')
    logepoch_text.append('Current RMSE [eV]: ' + str(rmse_to_log) + ', best point: ' + str(scheduler_to_log.best) +
                         ' , learning rate: ' + str(optimizer_tolog.param_groups[0]['lr']) + '\n\n')

    # Saving state_dictionaries to checkpoint
    torch.save({
        'model_state_dictionary': model_tolog.state_dict(),
        'optimizer_state_dictionary': optimizer_tolog.state_dict(),
        'scheduler_state_dictionary': scheduler_to_log.state_dict()
    }, save_to_name)

    # Saving log info to file
    with open(save_to_name.replace('.pth', '_log.txt'), 'w') as file:
        file.writelines(log_text)
    with open('epochslog.txt'.replace('epochs', 'epoch_' + day + '_' + month + '_' + year), 'a') as file2:
        file2.writelines(logepoch_text)


def validation_rmse(batches, model):
    """
    Computing RMSE for validation
    :param batches: chosen set of validation batches
    :param model: which model is used
    :return: float
    """
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_rmse = 0.0
    count = 0
    model.eval()
    for validation_batch in batches:
        curr_val_N = validation_batch.N
        prediction_internal_validation = model.energy_and_forces(Z=validation_batch.Z,
                                                                 Q=validation_batch.Q,
                                                                 S=validation_batch.S,
                                                                 R=validation_batch.R,
                                                                 idx_i=validation_batch.idx_i,
                                                                 idx_j=validation_batch.idx_j,
                                                                 batch_seg=validation_batch.batch_seg,
                                                                 num_batch=curr_val_N)
        predicted_valenergy_tensor = prediction_internal_validation[0]
        predicted_valforces_tensor = prediction_internal_validation[1]
        predicted_valdipmom_tensor = prediction_internal_validation[2]
        rmse, e_val, f_val, d_val = loss_function(energy_calc_tensor=predicted_valenergy_tensor,
                                                  energy_pred_tensor=validation_batch.E,
                                                  gradients_calc_tensor=validation_batch.F,
                                                  forces_pred_tensor=predicted_valforces_tensor,
                                                  dipole_calc_tensor=validation_batch.U,
                                                  dipole_pred_tensor=predicted_valdipmom_tensor,
                                                  molecule_amount=curr_val_N,
                                                  atom_permolecule_to_atom=validation_batch.loss_helper,
                                                  forces_bool=True, dipole_bool=False)
        """ Add to the total RMSE, divided by amount of batches later"""
        total_rmse += rmse * curr_val_N
        count += curr_val_N

    return total_rmse/count


def validation_by_molecule(batches, model):
    """
    Computing values by molecule, used for dataset reduction
    :param batches: chosen set of validation batches
    :param model: which model is used
    :return:
    """
    model.eval()
    list_of_names = []
    list_of_energyMAE = []
    batch_count = 0
    for validation_batch in batches:
        curr_val_N = validation_batch.N
        prediction_internal_validation = model.energy_and_forces(Z=validation_batch.Z,
                                                                 Q=validation_batch.Q,
                                                                 S=validation_batch.S,
                                                                 R=validation_batch.R,
                                                                 idx_i=validation_batch.idx_i,
                                                                 idx_j=validation_batch.idx_j,
                                                                 batch_seg=validation_batch.batch_seg,
                                                                 num_batch=curr_val_N)
        predicted_valenergy_tensor = prediction_internal_validation[0]
        predicted_valforces_tensor = prediction_internal_validation[1]
        predicted_valforcesvector_tensor = torch.sqrt(torch.sum(predicted_valforces_tensor ** 2, dim=1))
        energy_MAE = predicted_valenergy_tensor - validation_batch.E
        forces_MAE = predicted_valforcesvector_tensor + torch.sqrt(torch.sum(validation_batch.F, dim=1))
        list_of_names.extend(validation_batch.names)
        list_of_energyMAE.extend(energy_MAE.tolist())
        batch_count += 1

        print('Batch done :' + str(batch_count))

    return list_of_names, list_of_energyMAE


def loss_function(energy_pred_tensor=torch.FloatTensor([0]),
                  energy_calc_tensor=torch.FloatTensor([0]),
                  forces_pred_tensor=torch.FloatTensor([0]),
                  gradients_calc_tensor=torch.FloatTensor([0]),
                  dipole_pred_tensor=torch.FloatTensor([0]),
                  dipole_calc_tensor=torch.FloatTensor([0]),
                  molecule_amount=torch.FloatTensor([0]),
                  atom_permolecule_to_atom=torch.FloatTensor([0]),
                  forces_bool=False,
                  energies_bool=True,
                  dipole_bool=False,
                  forces_dot_bool=False,
                  type_forces='bymolecule',
                  e_weight=1,
                  f_weight=100,
                  q_weight=1):
    """
    Function calculating custom loss for training, can use sum of multiple loss terms.
    Assumes .forward() is used for training and .energy_and_forces() is used for validation
    :param energy_pred_tensor: [N] tensor of energies from .forward() or .energy_and_forces()
    :param energy_calc_tensor: [N] tensor of ground truth energies
    :param forces_pred_tensor: [N, 3] tensor of forces from .forward() or .energy_and_forces()
    :param gradients_calc_tensor: [N, 3] tensor of ground truth gradients (which are, by definition, force=-grad)
    :param dipole_pred_tensor: [N, 3] tensor of dipole moments from .forward() or .energy_and_forces()
    :param dipole_calc_tensor: [N] tensor of ground truths for norm of dipole moments
    :param molecule_amount: amount of molecules in batch
    :param atom_permolecule_to_atom: [N] tensor of atom count per molecule assigned to each atom, for forces loss
    :param forces_bool: if using forces for loss
    :param energies_bool: if using energies for loss
    :param dipole_bool: if using dipole moments for loss - IMPORTANT: for the first backward pass it always need to be False
    :param forces_dot_bool: if using a special function penalizing wrong direction of the force acting on atom
    :param type_forces: byatom or bymolecule, specifies if the mean of the forces is first calculated for each molecule
    :param e_weight: energy loss weight
    :param f_weight: forces loss weight
    :param q_weight: dipole moment loss weight
    :return:
    """
    energy_loss = 0
    forces_loss = 0
    dipole_loss = 0
    if energies_bool:
        energy_loss = torch.sqrt(sum((energy_pred_tensor - energy_calc_tensor)**2)/(len(energy_pred_tensor)))
        energy_loss = e_weight * energy_loss

    if forces_bool and type_forces == 'byatom':

        """ Subtracting ground truths from predictions, 
        then summing by the short dimension of squared 
        elements and sqrt to obtain loss norm vector per atom"""
        forces_lossnorm_per_atom = torch.sqrt(torch.sum((forces_pred_tensor + gradients_calc_tensor)**2, dim=1))

        """sqrt of mean per atom loss value"""
        forces_loss_per_batch = torch.sqrt(sum(forces_lossnorm_per_atom**2)/len(forces_lossnorm_per_atom))
        forces_loss = f_weight * forces_loss_per_batch

    if forces_bool and type_forces == 'bymolecule':

        """ Subtracting ground truths from predictions, 
        then summing by the short dimension of squared 
        elements and sqrt to obtain loss norm vector per atom"""
        forces_lossnorm_per_atom = torch.sum(((forces_pred_tensor + gradients_calc_tensor)**2), dim=1)

        """ Equation L = sqrt(sumBsum(a/BNi)"""
        b_times_N = atom_permolecule_to_atom*molecule_amount
        norms_divided = forces_lossnorm_per_atom/b_times_N
        forces_loss = f_weight * (torch.sqrt(torch.sum(norms_divided)))

    if dipole_bool:
        vectored_dipmom_pred = torch.sqrt(torch.sum(dipole_pred_tensor**2, dim=1))
        dipole_loss = torch.sqrt(sum((vectored_dipmom_pred - dipole_calc_tensor) ** 2) / (len(dipole_pred_tensor)))
        dipole_loss = q_weight * dipole_loss

    if forces_dot_bool:
        forces_pred_norm = torch.sqrt(torch.sum(forces_pred_tensor**2, dim=1))
        forces_calc_norm = torch.sqrt(torch.sum((-gradients_calc_tensor) ** 2, dim=1))
        K = forces_pred_norm * forces_calc_norm

        # J = torch.dot(forces_pred_tensor, (-gradients_calc_tensor))
        J = torch.einsum('nm,nm->n', forces_pred_tensor, (-gradients_calc_tensor))

        dot_loss1 = (torch.sum((K - J)))/len(K)
        '''ver 2, no comparing to the norm of vectors, just the'''
        dot_loss2 = torch.exp(torch.sum(-(torch.einsum('nm,nm->n', forces_pred_tensor, (-gradients_calc_tensor)))/K))

    loss = energy_loss + forces_loss + dipole_loss + dot_loss2

    return loss, energy_loss, forces_loss, dipole_loss
