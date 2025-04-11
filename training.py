import torch
from spookynet import SpookyNet
import math
import random
import json
import time
import datetime


class SpookyNetBatch:

    def __init__(self, device='cpu'):
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
        self.E = torch.tensor(self.E, dtype=torch.float32, device=self.device)
        self.Q = torch.tensor(self.Q, dtype=torch.float32, device=self.device)
        self.idx_i = torch.tensor(self.idx_i, dtype=torch.int64, device=self.device)
        self.idx_j = torch.tensor(self.idx_j, dtype=torch.int64, device=self.device)
        self.batch_seg = torch.tensor(self.batch_seg, dtype=torch.int64, device=self.device)
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
                 validation_set_to_log, save_to_name='checkpoint.pth', n_batch=100):
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
    :param model: which model we are using
    :return: float
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

    return math.sqrt(total_mse / count)


class TrainingApp:
    """
    Training app for the use with SpookyNet. To use the functionality it is best to use the .load_config() method and
    the entire program is initialized, then only the .train_model() functionality is required. WARNING! .load_config()
    is case-sensitive to the input file contents.

    Input file commands:
    dataset = <path to dataset> ;
    batch_size = <integer>  - size of a single batch;
    initial_lr = <float>  - initial learning rate;
    number_epochs = <integer>  - amount of epochs for training;
    save_to = <path or name>  - name for a produced model;
    load_checkpoint = <path or name>  - name for a produced checkpoint file;
    train_percent = <float between 0 and 100> - how much of the dataset will be used to train the model;

    Maybe will add some more input options later.
    """

    def __init__(self,
                 dataset=None,
                 model_name='best.pth',
                 start_lr=0.001,
                 num_epochs=10,
                 load_checkpoint=False,
                 checkpoint='checkpoint.pth',
                 batch_size=64,
                 molecules_loaded=False,
                 molecules=None,
                 batches_loaded=False,
                 all_batches = [],
                 model_initialized=False,
                 current_model=None,
                 optimizer=None,
                 scheduler=None,
                 split_stat=8,
                 set_device='cpu'):
        """

        Args:
            dataset: (str) path to dataset
            model_name: (str) name or path for a model trained using app
            start_lr:  (float) starting learning rate
            num_epochs: (int) number of epochs
            load_checkpoint: (bool) flag for loading checkpoint from file
            checkpoint: (str) name or path for checkpoint to load
            batch_size: (int) amount of molecules in a single batch
            molecules_loaded: (bool) checks if molecules are loaded
            molecules: (list) list of all molecules
            batches_loaded: (bool) checks if batches are loaded
            all_batches: (list) list of all batches saved to SpookyBatch class objects
            model_initialized: (bool) checks if model is initialized
            current_model: (torch.model) SpookyNet by default, better not change
            optimizer: (torch.optimizer) Adam  by default, better not change, maybe
            will add functionality to customize later
            scheduler: (torch.scheduler) ReduceLROnPlateau by default, better not change, maybe
            will add functionality to customize later
        """

        self.dataset = dataset
        self.model_name = model_name
        self.start_lr = start_lr
        self.num_epochs = num_epochs
        self.load_checkpoint = load_checkpoint
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.molecules_loaded = molecules_loaded
        self.molecules = molecules
        self.batches_loaded = batches_loaded
        self.all_batches = all_batches
        self.model_initialized = model_initialized
        self.current_model = current_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.split_stat = split_stat
        self.set_device = set_device

    def load_config(self, config_path):
        """
        Method loading input file to be used as configuration tool for the training process. After loading the
        configuration parameters initialises the molecules, batches, model, optimizer and scheduler. WARNING do not use
        .load_molecules() .load_batches() and .initialize_model() methods with .load_config()!!!!!
        Args:
            config_path: path to the config input file
        """
        with open(config_path, 'r') as file:
            content = file.readlines()
        for line in content:
            if 'dataset' in line:
                working_line = line.split(' ')
                self.dataset = str(working_line[2].replace('\n', ''))
            if 'device' in line:
                working_line = line.split(' ')
                self.set_device = str(working_line[2].replace('\n', ''))
                if self.set_device != 'cpu' and self.set_device != 'cuda':
                    raise Exception('Wrong device, can only use cuda or cpu!')
            if 'batch_size' in line:
                working_line = line.split(' ')
                self.batch_size = int(working_line[2].replace('\n', ''))
            if 'initial_lr' in line:
                working_line = line.split(' ')
                self.start_lr = float(working_line[2].replace('\n', ''))
            if 'save_to' in line:
                working_line = line.split(' ')
                self.model_name = str(working_line[2].replace('\n', ''))
            if 'number_epochs' in line:
                working_line = line.split(' ')
                self.num_epochs = int(working_line[2].replace('\n', ''))
            if 'load_checkpoint' in line:
                working_line = line.split(' ')
                self.load_checkpoint = True
                self.checkpoint = str(working_line[2].replace('\n', ''))
            if 'train_percent' in line:
                working_line = line.split(' ')
                self.split_stat = float(working_line[2].replace('\n', ''))
        self.load_molecules(self.dataset)
        self.load_batches()
        self.initialize_model()

    def load_molecules(self, path_dataset):
        """
        Method loading dictionary from the specified database and converting them to be digestible by the NN. If loaded
        from checkpoint the model will use molecules.json created alongside the checkpoint file to ensure the training
        and validation datasets are not changed.
        Args:
            path_dataset: (str) path to the stored dataset
        """
        if not self.molecules_loaded:

            # Loading from database if starting without checkpoint
            if not self.load_checkpoint:
                with open(path_dataset, 'r') as file_internal:
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
                    energy = float(entry.get('Energy [Hartree]')) * 27.2107
                    entry['Energy [eV] (float)'] = energy
                random.shuffle(lines)
                self.molecules = lines
                self.molecules_loaded = True
                print('Molecules loaded')

            # Loading from previous state if using checkpoint
            # TODO check for compatibility!!!!
            if self.load_checkpoint:
                with open('molecules.json', 'r') as file_internal:
                    lines = json.load(file_internal)
                self.molecules = lines
                self.molecules_loaded = True
                print('Molecules loaded')

        elif self.molecules_loaded:
            raise Exception("Molecules already loaded, use .clear_molecules()")

    def load_batches(self):
        """
        Transforms loaded molecules to the list of SpookyBatch objects of specified size
        """
        if not self.batches_loaded and self.molecules_loaded:
            batch = None
            number_molecules = 0  # how many molecules already loaded into the current batch
            # number_atoms = 0  # total number of atoms in this batch
            for molecule in self.molecules:
                if number_molecules == 0:
                    number_atoms = 0
                    batch = SpookyNetBatch(device=self.set_device)  #stores the data in a format we can pass to SpookyNet

                batch.Z.extend(molecule['Atomic number'])
                batch.R.extend(molecule['Position of atom (float)'])
                batch.Q.append(molecule['Charge (int)'])
                batch.E.append(molecule['Energy [eV] (float)'])
                cur_idx_i, cur_idx_j = get_idx(molecule['Position of atom (float)'])
                cur_idx_i += number_atoms
                cur_idx_j += number_atoms
                batch.idx_i.extend(cur_idx_i)
                batch.idx_j.extend(cur_idx_j)
                batch.batch_seg.extend([number_molecules] * len(molecule['Atomic number']))
                number_atoms += len(molecule['Atomic number'])
                number_molecules += 1

                if number_molecules >= self.batch_size:
                    batch.N = number_molecules
                    self.all_batches.append(batch.toTensor())
                    number_molecules = 0
                    batch = None

            if batch:
                batch.N = number_molecules
                self.all_batches.append(batch.toTensor())
            self.batches_loaded = True
            print('Batches loaded')
        elif self.batches_loaded:
            raise Exception("Batches already loaded, to reinitialize batches use .clear_batches()!")
        elif not self.molecules_loaded:
            raise Exception("Molecules not loaded, use .load_molecules()!")

    def initialize_model(self):
        """
        Initializes model, optimizer and scheduler
        """
        if not self.model_initialized:
            if self.set_device == 'cpu':
                self.current_model = SpookyNet().to(torch.float32).cpu()
            elif self.set_device == 'cuda':
                self.current_model = SpookyNet().to(torch.float32).cuda()
            self.optimizer = torch.optim.Adam(self.current_model.parameters(), lr=self.start_lr, amsgrad=True)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5,
                                                                        patience=25, threshold=0)
            self.model_initialized = True
        elif self.model_initialized:
            raise Exception("Model already initialized!")

    def train_model(self):
        """
        Trains the model using specified hyperparameters, needs batches to be loaded; needs model, optimizer and
        scheduler to be initialized.
        """
        if self.batches_loaded and self.model_initialized:

            # Saving set to be used for checkpoint reinitialization
            '''
            if not self.load_checkpoint:
                with open('molecules.json', 'w') as file:
                    json.dump(self.molecules, file, ensure_ascii=False, indent=4)
            '''
            set_size = len(self.all_batches)
            training_endpoint = (set_size // 10) * self.split_stat
            training = self.all_batches[:int(training_endpoint)]
            validation = self.all_batches[int(training_endpoint):]

            if self.load_checkpoint:
                checkpoint_state = torch.load(self.checkpoint)
                self.optimizer.load_state_dict(checkpoint_state['optimizer_state_dictionary'])
                self.current_model.load_state_dict(checkpoint_state['model_state_dictionary'])
                self.scheduler.load_state_dict(checkpoint_state['scheduler_state_dictionary'])
                print('Loaded from checkpoint - Epoch {}: RMSE for best [eV]: {:.3f}'.format(
                    self.scheduler.last_epoch, self.scheduler.best))

            mse_sum = torch.nn.MSELoss(reduction='sum')

            self.current_model.train()

            for epoch in range(self.num_epochs):
                start_epoch = time.perf_counter()
                random.shuffle(training)

                # Iterating through batches
                current_iteration = 0
                for training_batch in training:
                    current_iteration += 1
                    curr_train_N = training_batch.N
                    res = self.current_model.energy(Z=training_batch.Z, Q=training_batch.Q, S=training_batch.S,
                                                    R=training_batch.R,
                                                    idx_i=training_batch.idx_i, idx_j=training_batch.idx_j,
                                                    batch_seg=training_batch.batch_seg, num_batch=curr_train_N)
                    energy_tensor = res[0]

                    # Defining loss
                    loss = mse_sum(energy_tensor, training_batch.E) / curr_train_N
                    if current_iteration % 1000 == 0:
                        loss_text, curr_step, curr_epoch = loss, current_iteration, epoch
                        print(f"Training loss: {loss_text:>7f}  step/epoch [{curr_step:>5d}/{curr_epoch:>5d}]")
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()



                # Validation
                rmse = validation_rmse(validation, self.current_model)
                if self.scheduler.is_better(rmse, self.scheduler.best):
                    self.current_model.save(self.model_name)
                    logging_data(self.current_model, self.optimizer, self.scheduler, rmse, training, validation,
                                 save_to_name='checkpoint_best.pth', n_batch=self.batch_size)
                self.current_model.train()
                self.scheduler.step(rmse)

                # Saving checkpoint
                logging_data(self.current_model, self.optimizer, self.scheduler, rmse, training, validation)

                end_epoch = time.perf_counter()
                print(
                    'Epoch {}: Validation loss [eV]: {:.3f}, for best [eV]: {:.3f}, learning rate: {:.6f},  done in {:.1f} seconds'.format(
                        self.scheduler.last_epoch, rmse, self.scheduler.best, self.optimizer.param_groups[0]['lr'],
                        (end_epoch - start_epoch)))

        elif not self.batches_loaded:
            raise Exception("Batches not loaded!")
        elif not self.model_initialized:
            raise Exception("Model not initialized!")


# !!!!!!!!!!!! TRAINING TEST !!!!!!!!!!!!
if __name__ == "__main__":
    start_program_time = time.perf_counter()
    training_functionality = TrainingApp()
    training_functionality.load_config('./config.inp')
    training_functionality.train_model()
    end_program_time = time.perf_counter()
    print('Done in ' + str(end_program_time - start_program_time) + 's')
