import torch
from spookynet import SpookyNet
import math
import random
import json
import time
import datetime
import numpy as np
from utility_training import *
import statistics


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
    saved_order = <path or name> - name of a molecule order file generated in a training run
    remove_atomrefs - removes computed atomization energies for specified atom types in a given level of theory
    (if using anything other than wb97M-V with def2-TZVP the values in lines 194-209 need to be changed accordingly)

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
                 all_batches=[],
                 model_initialized=False,
                 current_model=None,
                 optimizer=None,
                 scheduler=None,
                 split_stat=80,
                 set_device='cpu',
                 order_savedstate=None):
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
            split_stat: (int) how many percent of the set will be used as training set, between 0-100
            set_device: (str) what device should be used for training, either cpu or cuda
            order_savedstate: (str) name or path to the molecules order file

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
        self.order_savedstate = order_savedstate
        self.molecules_order = []
        self.remove_atomrefs = False
        self.remove_mean = False
        self.center_molecules = True
        self.mean_atom = 0
        self.std_dev = 1
        self.d4_bool = True
        self.optimizer_choice = "Adam"

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
                self.split_stat = (float(working_line[2].replace('\n', ''))) / 10
            if 'saved_order' in line:
                working_line = line.split(' ')
                self.order_savedstate = str(working_line[2].replace('\n', ''))
            if 'remove_atomrefs' in line:
                self.remove_atomrefs = True
            if 'remove_mean' in line:
                self.remove_mean = True

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
        start_molload = time.perf_counter()
        if not self.molecules_loaded:

            """Loading database"""
            with open(path_dataset, 'r') as file_internal:
                lines = json.load(file_internal)

            internal_molecules_order = []
            new_lines = []
            """Determining molecule order, either as in datafile or from saved_order"""
            if self.order_savedstate is None:
                """If no save_order file is given"""
                for x in range(len(lines)):
                    internal_molecules_order.append(x + 1)
            else:
                # If save_order file is given
                with open(self.order_savedstate, 'r') as file:
                    molecules_order_string = file.read()
                    internal_molecules_order = molecules_order_string.split('->')
                    internal_molecules_order.pop()

            """Revised version of converting database values, account for loading from predefined molecules list"""
            sum_energies = 0
            sum_molecules = 0
            energies = []
            for x in internal_molecules_order:
                entry = lines[int(x) - 1]
                atomic_number = []
                ref_energy = 0

                """Conversion table from atom types to atomic numbers, may streamline with dict later"""
                atom_types_internal = entry.get('Type of atom')
                for single_type_internal in atom_types_internal:
                    if single_type_internal == 'H':
                        atomic_number.append(1)
                        ref_energy += (-0.494116047851 * 27.2107)
                    elif single_type_internal == 'C':
                        atomic_number.append(6)
                        ref_energy += (-37.787880174304 * 27.2107)
                    elif single_type_internal == 'O':
                        atomic_number.append(8)
                        ref_energy += (-74.982757125171 * 27.2107)
                    elif single_type_internal == 'N':
                        atomic_number.append(7)
                        ref_energy += (-54.489699377480 * 27.2107)
                    elif single_type_internal == 'P':
                        atomic_number.append(15)
                        ref_energy += (-341.181628271541 * 27.2107)
                entry['Atomic number'] = atomic_number
                entry['Reference energy'] = ref_energy

                """Converting positions to float"""
                float_positions = []
                position_lines = entry.get('Position of atom')

                """Layers, from list of 3 positions to single position"""
                for triple in position_lines:
                    triplets_positions = []
                    for single in triple:
                        triplets_positions.append(float(single))
                    float_positions.append(triplets_positions)
                entry['Position of atom (float)'] = float_positions

                """Centering of the molecule to obtain the uniform scheme, as denoted in SpookyNet paper
                (important for charged molecules)"""
                if self.center_molecules:
                    points = entry.get('Position of atom (float)')
                    center = [statistics.mean(i) for i in zip(*points)]
                    centered_positions = []
                    for point in points:
                        a = point[0] - center[0]
                        b = point[1] - center[1]
                        c = point[2] - center[2]
                        centered_positions.append([a, b, c])
                    entry['Position of atom (float)'] = centered_positions

                """Converting gradients to float, and a.u to eV/Angstrom"""
                float_gradients = []
                gradients_lines = entry.get('Gradients')

                """Same as postions"""
                for triple in gradients_lines:
                    triplets_gradients = []
                    for single in triple:
                        triplets_gradients.append(float(single) * 51.42)
                    float_gradients.append(triplets_gradients)
                entry['Gradients [eV/Ang] (float)'] = float_gradients

                """Converting charges to integers, may be redundant"""
                charge = int(entry.get('Charge'))
                entry['Charge (int)'] = charge

                dipole_moment = float(entry.get('Dipole moment'))
                entry['Dipole moment'] = dipole_moment * 0.52918

                """Converting Energies to floats, changing to eV, adding offsets"""
                energy = float(entry.get('Energy [Hartree]')) * 27.2107
                entry['Energy [eV] (float)'] = energy
                # for atomrefs alone
                if self.remove_atomrefs:
                    entry['Energy reduced [eV] (float)'] = energy - entry.get('Reference energy')
                    energies.append(entry['Energy reduced [eV] (float)'])
                else:
                    energies.append(entry['Energy [eV] (float)'])
                # for atomrefs and mean
                if self.remove_mean and self.remove_atomrefs:
                    reduced_energy = entry.get('Energy reduced [eV] (float)') / int(entry.get("Size [atoms]"))
                    sum_energies += reduced_energy
                # for mean alone
                if self.remove_mean and not self.remove_atomrefs:
                    reduced_energy = entry.get('Energy [eV] (float)') / int(entry.get("Size [atoms]"))
                    sum_energies += reduced_energy

                loss_forces_atom_count = []
                for t in range(0, len(entry['Atomic number'])):
                    loss_forces_atom_count.append(len(entry['Atomic number']))
                entry['Atom count vector for loss'] = loss_forces_atom_count

                sum_molecules += 1
                new_lines.append(entry)
            self.mean_atom = sum_energies / sum_molecules
            self.std_dev = np.std(energies)

            ''' Shuffle to randomize the set '''
            if self.order_savedstate is None:
                random.shuffle(new_lines)

            ''' This is to save ordering of in the case of resuming training from checkpoint '''
            for entry in new_lines:
                entry_number = int(entry.get('Entry number'))
                self.molecules_order.append(str(entry_number))
                self.molecules_order.append(str('->'))
            if self.order_savedstate is None:
                with open('molecule_order.txt', 'w') as file:
                    file.writelines(self.molecules_order)

            self.molecules = new_lines
            self.molecules_loaded = True
            end_molload = time.perf_counter()
            print('Molecules loaded in  {:.1f} seconds'.format((end_molload - start_molload)))

        elif self.molecules_loaded:
            raise Exception("Molecules already loaded, use .clear_molecules()")

    def load_batches(self):
        """
        Transforms loaded molecules to the list of SpookyBatch objects of specified size
        """
        if not self.batches_loaded and self.molecules_loaded:
            start_batchload = time.perf_counter()
            batch = None
            number_molecules = 0  # how many molecules already loaded into the current batch
            # number_atoms = 0  # total number of atoms in this batch
            for molecule in self.molecules:
                if number_molecules == 0:
                    number_atoms = 0
                    batch = SpookyNetBatch(
                        device=self.set_device)  #stores the data in a format we can pass to SpookyNet

                batch.Z.extend(molecule['Atomic number'])
                batch.R.extend(molecule['Position of atom (float)'])
                batch.F.extend(molecule['Gradients [eV/Ang] (float)'])
                batch.loss_helper.extend(molecule['Atom count vector for loss'])
                batch.Q.append(molecule['Charge (int)'])
                batch.U.append(molecule['Dipole moment'])
                batch.names.append(molecule['Entry number'])
                # Alternate versions for reduced and not, mean is always subtracted because if
                # self.remove_mean is False, the mean is 0, thus not affecting energy
                if self.remove_atomrefs:
                    batch.E.append(
                        (molecule['Energy reduced [eV] (float)'] - (self.mean_atom * int(molecule["Size [atoms]"]))))
                else:
                    batch.E.append((molecule['Energy [eV] (float)'] - (self.mean_atom * int(molecule["Size [atoms]"]))))
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
            end_batchload = time.perf_counter()
            print('Batches loaded in  {:.1f} seconds'.format((end_batchload - start_batchload)))
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
                self.current_model = SpookyNet(use_d4_dispersion=self.d4_bool, compute_d4_atomic=self.d4_bool).to(torch.float32).cpu()
            elif self.set_device == 'cuda':
                self.current_model = SpookyNet(use_d4_dispersion=self.d4_bool, compute_d4_atomic=self.d4_bool).to(torch.float32).cuda()
            if self.optimizer_choice == 'Adam':
                self.optimizer = torch.optim.Adam(self.current_model.parameters(),
                                                  lr=self.start_lr,
                                                  amsgrad=True)
            elif self.optimizer_choice == 'SGD':
                self.optimizer = torch.optim.SGD(self.current_model.parameters(),
                                                 lr=self.start_lr,
                                                 nesterov=True,
                                                 momentum=0.9)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                        factor=0.5,
                                                                        patience=25,
                                                                        threshold=0)
            self.model_initialized = True
            print('Model on cuda: ' + str(next(self.current_model.parameters()).is_cuda))
            print('Model initialized')
        elif self.model_initialized:
            raise Exception("Model already initialized!")

    def train_model(self):
        """
        Trains the model using specified hyperparameters, needs batches to be loaded; needs model, optimizer and
        scheduler to be initialized.
        """
        if self.batches_loaded and self.model_initialized:

            set_size = len(self.all_batches)
            training_endpoint = (set_size // 10) * self.split_stat
            training = self.all_batches[:int(training_endpoint)]
            validation = self.all_batches[int(training_endpoint):]
            start = True

            if self.load_checkpoint:
                checkpoint_state = torch.load(self.checkpoint)
                self.optimizer.load_state_dict(checkpoint_state['optimizer_state_dictionary'])
                self.current_model.load_state_dict(checkpoint_state['model_state_dictionary'])
                self.scheduler.load_state_dict(checkpoint_state['scheduler_state_dictionary'])
                print('Loaded from checkpoint - Epoch {}: RMSE for best [eV]: {:.3f}'.format(
                    self.scheduler.last_epoch, self.scheduler.best))
                start = False

            self.current_model.train()

            for epoch in range(self.num_epochs):

                start_epoch = time.perf_counter()

                """ Shuffling training set on each epoch, prevents pattern from batch order"""
                random.shuffle(training)

                """Iterating through batches"""
                current_iteration = 0

                for training_batch in training:
                    """ Zeroing gradients """
                    self.optimizer.zero_grad()

                    current_iteration += 1
                    curr_train_N = training_batch.N

                    """ Forward pass to obtain guess"""
                    prediction_internal_results = self.current_model.forward(Z=training_batch.Z,
                                                                             Q=training_batch.Q,
                                                                             S=training_batch.S,
                                                                             R=training_batch.R,
                                                                             idx_i=training_batch.idx_i,
                                                                             idx_j=training_batch.idx_j,
                                                                             batch_seg=training_batch.batch_seg,
                                                                             num_batch=curr_train_N)

                    """ Obtaining predicted tensors from batch"""
                    predicted_energy_tensor = prediction_internal_results[0]
                    predicted_forces_tensor = prediction_internal_results[1]
                    predicted_dipmom_tensor = prediction_internal_results[2]

                    """ This needs to be done because and the begining the network 
                    is not giving any result for dipole moments. Considering this, 
                    the backwards pass gives NaNs in second pass"""
                    if start:
                        for_dipole_bool = False
                    else:
                        for_dipole_bool = True

                    """ Obtaining loss """
                    loss, e_loss, f_loss, d_loss = loss_function(energy_calc_tensor=predicted_energy_tensor,
                                                                 energy_pred_tensor=training_batch.E,
                                                                 gradients_calc_tensor=training_batch.F,
                                                                 forces_pred_tensor=predicted_forces_tensor,
                                                                 dipole_calc_tensor=training_batch.U,
                                                                 dipole_pred_tensor=predicted_dipmom_tensor,
                                                                 molecule_amount=curr_train_N,
                                                                 atom_permolecule_to_atom=training_batch.loss_helper,
                                                                 forces_bool=True, dipole_bool=for_dipole_bool)
                    if current_iteration % self.batch_size*10 == 0:
                        loss_text, curr_step, curr_epoch, e_text, f_text, d_text = loss, current_iteration, epoch, e_loss, f_loss, d_loss
                        print(f"Training loss: {loss_text:>6f}, dE={e_text:>6f}, dF={f_text:>6f} dU={d_text:>6f} step/epoch [{curr_step:>5d}/{curr_epoch:>5d}]")
                    '''loss_text, curr_step, curr_epoch, e_text, f_text, d_text = loss, current_iteration, epoch, e_loss, f_loss, d_loss
                    print(f"Training loss: {loss_text:>6f}, dE={e_text:>6f}, dF={f_text:>6f} dU={d_text:>6f} step/epoch [{curr_step:>5d}/{curr_epoch:>5d}]")'''

                    loss.backward()
                    if self.d4_bool:
                        self.current_model.d4_dispersion_energy._compute_refc6()
                    self.optimizer.step()
                    start = False

                """ Validation, chosen each epoch for validation, may change to stepwise vaildation"""
                rmse = validation_rmse(validation, self.current_model)
                if self.scheduler.is_better(rmse, self.scheduler.best):
                    self.current_model.save(self.model_name)
                    logging_data(self.current_model,
                                 self.optimizer,
                                 self.scheduler,
                                 rmse,
                                 training,
                                 validation,
                                 save_to_name='checkpoint_best.pth',
                                 n_batch=self.batch_size)

                """ Re-enabling training mode """
                self.current_model.train()
                self.scheduler.step(rmse)

                """ Saving checkpoint """
                logging_data(self.current_model,
                             self.optimizer,
                             self.scheduler,
                             rmse,
                             training,
                             validation,
                             mean_tolog=self.mean_atom)

                end_epoch = time.perf_counter()
                print(
                    'Epoch {}: Validation loss [eV]: {:.3f}, for best [eV]: {:.3f}, learning rate: {:.6f},  done in {:.1f} seconds'.format(
                        self.scheduler.last_epoch, rmse, self.scheduler.best, self.optimizer.param_groups[0]['lr'],
                        (end_epoch - start_epoch)))

            """ Sanity check, raises exception if anything has not been loaded properly """
        elif not self.batches_loaded:
            raise Exception("Batches not loaded!")
        elif not self.model_initialized:
            raise Exception("Model not initialized!")

    def return_batches(self):
        return self.all_batches

    def return_model(self):
        return self.current_model


# !!!!!!!!!!!! TRAINING TEST !!!!!!!!!!!!
if __name__ == "__main__":
    start_program_time = time.perf_counter()
    training_functionality = TrainingApp()
    print('App initialized.')
    training_functionality.load_config('./config.inp')
    print('Config loaded, starting training procedure.')
    training_functionality.train_model()
    end_program_time = time.perf_counter()
    print('Full training according to specs done in ' + str(end_program_time - start_program_time) + 'seconds.')
    with open('quotes.txt', 'r') as quotes:
        quote_list = quotes.readlines()
    choose_quote = random.randrange(0, len(quote_list))
    print(quote_list[choose_quote])
