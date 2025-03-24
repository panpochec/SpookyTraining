import torch
from spookynet import SpookyNet
import math
import random
import json
import time
import datetime


class SpookyBatch:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self):
        """
        Initializing class with internal parameters for class set as (mimics SpookyNet):
        N: numbers of molecules in a batch
        Z: atomic numbers of atoms
        R: positions of atoms
        E: energies of molecules
        Q: charges of molecules
        S: spins for molecules (not using for now)
        batch_seg: utility for assigning Z and R to molecules

        """
        self.N = 0
        self.Z = []
        self.R = []
        self.E = []
        self.Q = []
        self.S = []
        self.batch_seg = []
        self.idx_i = []
        self.idx_j = []

    def toTensor(self):
        """
        Translates internal parameters to tensors
        :return:
        """
        self.S = torch.zeros(self.N, dtype=torch.float32, device=SpookyBatch.device)  # not using spins for now, hence zeroes
        self.Z = torch.tensor(self.Z, dtype=torch.int64, device=SpookyBatch.device)
        self.R = torch.tensor(self.R, dtype=torch.float32, device=SpookyBatch.device, requires_grad=True)
        self.E = torch.tensor(self.E, dtype=torch.float32, device=SpookyBatch.device)
        self.Q = torch.tensor(self.Q, dtype=torch.float32, device=SpookyBatch.device)
        self.idx_i = torch.tensor(self.idx_i, dtype=torch.int64, device=SpookyBatch.device)
        self.idx_j = torch.tensor(self.idx_j, dtype=torch.int64, device=SpookyBatch.device)
        self.batch_seg = torch.tensor(self.batch_seg, dtype=torch.int64,  device=SpookyBatch.device)
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


def load_molecules(path, type_data='json_database'):
    """
    Function loading parameters from the specified database and converting them to be digestible by the NN

    :param path: path to database file, can be relative or absolute
    :type path: str
    :param type_data: no use dor now - possible future expansion of db types
    :type type_data: str
    :return:
    """
    if type_data == 'json_database':
        with open(path, 'r') as file_internal:
            lines = json.load(file_internal)

        # Conversion table from atom types to atomic numbers, may streamline with dict later
        # lines - whole database, entry - dictionary for a single molecule
        for entry in lines:
            atomic_number = []
            atom_types_internal = entry.get('Type of atom')
            for single_type_internal in atom_types_internal:
                if single_type_internal == 'H':
                    atomic_number.append(1)
                elif single_type_internal == 'C':
                    atomic_number.append(6)
                elif single_type_internal == 'O':
                    atomic_number.append(8)
                elif single_type_internal == 'N':
                    atomic_number.append(7)
                elif single_type_internal == 'P':
                    atomic_number.append(15)
            entry['Atomic number'] = atomic_number

        # Converting positions to float
        for entry in lines:
            float_positions = []
            position_lines = entry.get('Position of atom')

            # Layers, from list of 3 positions to single position
            for triple in position_lines:
                triplets_positions = []
                for single in triple:
                    triplets_positions.append(float(single))
                float_positions.append(triplets_positions)
            entry['Position of atom (float)'] = float_positions

        # Converting charges to integers, may be redundant
        for entry in lines:
            charge = int(entry.get('Charge'))
            entry['Charge (int)'] = charge

        # Converting Energies to floats, changing to eV
        for entry in lines:
            energy = float(entry.get('Energy [Hartree]'))*27.2107
            entry['Energy [eV] (float)'] = energy

    return lines


def load_batches(molecules, full_batch=100):
    """

    :param molecules:
    :type molecules: list
    :param full_batch: (optional) how many molecules should be loaded into a single batch
    :type full_batch: int
    :return:
    """
    all_batches = []
    batch = None
    nm = 0  # how many molecules already loaded into the current batch
    for molecule in molecules:
        if nm == 0:
            na = 0  # num total atoms in this batch
            batch = SpookyBatch()  # stores the data in a format we can pass to SpookyNet

        batch.Z.extend(molecule['Atomic number'])
        batch.R.extend(molecule['Position of atom (float)'])
        batch.Q.append(molecule['Charge (int)'])
        batch.E.append(molecule['Energy [eV] (float)'])
        cur_idx_i, cur_idx_j = get_idx(molecule['Position of atom (float)'])
        cur_idx_i += na
        cur_idx_j += na
        batch.idx_i.extend(cur_idx_i)
        batch.idx_j.extend(cur_idx_j)
        batch.batch_seg.extend([nm] * len(molecule['Atomic number']))
        na += len(molecule['Atomic number'])
        nm += 1

        if nm >= full_batch:
            batch.N = nm
            all_batches.append(batch.toTensor())
            nm = 0
            batch = None

    if batch:
        batch.N = nm
        all_batches.append(batch.toTensor())

    return all_batches


def logging_data(model_tolog, optimizer_tolog, scheduler_to_log, rmse_to_log, training_set_to_log,
                 validation_set_to_log, save_to_name='checkpoint.pth'):
    """

    :param model_tolog: model to log
    :param optimizer_tolog: optimizer to log
    :param scheduler_to_log: scheduler to log
    :param rmse_to_log: rmse of a logged point
    :param training_set_to_log: training set, used to extract the size of the sets
    :param validation_set_to_log:  validation set, used to extract the size of the sets
    :param save_to_name: name used for checkpoint file, checkpoint.pth by default
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
    log_text.append('Log data for ' + save_to_name + ', recorded on ' + logging_date + '\n')
    log_text.append('Optimizer: ' + str(type(optimizer_tolog).__name__) + ', epoch ' + str(scheduler_to_log.last_epoch)
                    + '\n')
    log_text.append('Current RMSE [eV]: ' + str(rmse_to_log) + ', best point: ' + str(scheduler_to_log.best) + '\n')
    log_text.append('Training set size: ' + str(len(training_set_to_log)*100) + '\n')
    log_text.append('Validation set size: ' + str(len(validation_set_to_log)*100) + '\n')

    logepoch_text = []
    logepoch_text.append('Epoch ' + str(scheduler_to_log.last_epoch) + ' recorded on ' + logging_date + '\n')
    logepoch_text.append('Current RMSE [eV]: ' + str(rmse_to_log) + ', best point: ' + str(scheduler_to_log.best) + '\n\n')

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


def training_function(dataset, model_name='best.pth', start_lr=0.001, num_epochs=10, load_checkpoint=False,
                      checkpoint='checkpoint.pth'):
    """
    Training function using SpookyNet to train a model

    :param dataset: path to dataset used to train and validate the NN, can be relative to script or absolute
    :type dataset: str
    :param model_name: path to the stored best point of the model
    :type model_name: str
    :param start_lr: (optional) custom starting learning rate for the scheduler, 0.001 by default
    :type start_lr: float
    :param num_epochs: (optional) number of epochs for training (disregards previous epochs if load_checkpoint=True),
    10 by default
    :type num_epochs: int
    :param load_checkpoint: (optional) specifies if a checkpoint for state_dict will be used, for retraining/resuming,
    False by default
    :type load_checkpoint: bool
    :param checkpoint: path to the checkpoint file
    :type checkpoint: str
    :return: None, saves model to best.pth file
    """

    # Loading data from dataset, shuffling entries
    start_test1 = time.perf_counter()
    molecules_test = load_molecules(dataset)
    random.shuffle(molecules_test)
    end_test1 = time.perf_counter()
    print('Loading data done in: {:.2f} seconds'.format((end_test1 - start_test1)))

    # Loading molecules to SpookyBatches
    molecules_test_batches = load_batches(molecules_test)
    random.shuffle(molecules_test_batches)
    set_size = len(molecules_test_batches)
    training_endpoint = (set_size//10) * 8

    # Assigning molecules to sets, should be done in a more elegant way - TODO later
    '''training = molecules_test_batches[:training_endpoint]
    validation = molecules_test_batches[training_endpoint:]'''
    training = molecules_test_batches[:21]
    validation = molecules_test_batches[21:23]

    end_test2 = time.perf_counter()
    print('Full data preparation done in: {:.2f} seconds'.format((end_test2 - start_test1)))

    print('Starting the training procedure:')

    best_point = model_name

    # Initializing NN
    current_model = SpookyNet().to(torch.float32).cpu()
    current_model.train()

    # Initializing for training
    optimizer = torch.optim.Adam(current_model.parameters(), lr=start_lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=25, threshold=0)

    # Loading a checkpoint file if specified, for retraining
    if load_checkpoint:
        checkpoint_state = torch.load(checkpoint)
        optimizer.load_state_dict(checkpoint_state['optimizer_state_dictionary'])
        current_model.load_state_dict(checkpoint_state['model_state_dictionary'])
        scheduler.load_state_dict(checkpoint_state['scheduler_state_dictionary'])
        print('Loaded from checkpoint - Epoch {}: RMSE for best [eV]: {:.3f}'.format(
            scheduler.last_epoch, scheduler.best))

    # Defining loss error assessment for loss
    mse_sum = torch.nn.MSELoss(reduction='sum')

    # Iterating through epochs
    for epoch in range(num_epochs):
        start_epoch = time.perf_counter()
        random.shuffle(training)

        # Iterating through batches
        for training_batch in training:
            curr_train_N = training_batch.N
            res = current_model.energy(Z=training_batch.Z, Q=training_batch.Q, S=training_batch.S, R=training_batch.R,
                               idx_i=training_batch.idx_i, idx_j=training_batch.idx_j,
                               batch_seg=training_batch.batch_seg, num_batch=curr_train_N)
            energy_tensor = res[0]

            # Defining loss
            loss = mse_sum(energy_tensor, training_batch.E)/curr_train_N

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        rmse = validation_rmse(validation, current_model)
        if scheduler.is_better(rmse, scheduler.best):
            current_model.save(best_point)
            logging_data(current_model, optimizer, scheduler, rmse, training, validation,
                         save_to_name='checkpoint_best.pth')
        current_model.train()
        scheduler.step(rmse)

        # Saving checkpoint
        logging_data(current_model, optimizer, scheduler, rmse, training, validation)

        end_epoch = time.perf_counter()
        print('Epoch {}: RMSE for current [eV]: {:.3f}, RMSE for best [eV]: {:.3f}, done in {:.1f} seconds'.format(
            scheduler.last_epoch, rmse, scheduler.best, (end_epoch-start_epoch)))


def validation_rmse(batches, model):
    """
    Computing RMSE for validation
    :param batches: chosen set of validation batches
    :param model: which model we are using
    :return:
    """
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    count = 0
    model.eval()
    for validation_batch in batches:
        curr_val_N = validation_batch.N
        res = model.energy(Z=validation_batch.Z, Q=validation_batch.Q, S=validation_batch.S, R=validation_batch.R,
                           idx_i=validation_batch.idx_i, idx_j=validation_batch.idx_j,
                           batch_seg=validation_batch.batch_seg, num_batch=curr_val_N)
        energy_tensor = res[0]

        # sum over pairings
        total_mse += mse_sum(energy_tensor, validation_batch.E).item()
        count += curr_val_N

    return math.sqrt(total_mse/count)


# !!!!!!!!!!!! TRAINING TEST !!!!!!!!!!!!
training_function('./dataset2.json', num_epochs=100, load_checkpoint=True)

print('done')