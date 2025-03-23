from ase import Atoms
from ase import io
from spookynet import SpookyNetCalculator
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.md.nvtberendsen import NVTBerendsen
from ase.optimize import BFGS


molecule = io.read('opt23.xyz', format='xyz')
# magmom is the number of unpaired electrons, i.e. 0 means singlet
molecule.calc = SpookyNetCalculator(load_from="best.pth", charge=0, magmom=0)
print("singlet energy")
print((molecule.get_potential_energy())/27.2107)
molecule.calc = SpookyNetCalculator(load_from="parameters.pth", charge=0, magmom=0)
print("singlet energy")
print(molecule.get_potential_energy())

# magmom=2 means 2 unpaired electrons => triplet state
'''carbene.set_calculator(SpookyNetCalculator(load_from="parameters.pth", charge=0, magmom=2))
print("triplet energy")
print(carbene.get_potential_energy())'''

'''dyn = NVTBerendsen(carbene, timestep=5.0 * units.fs, temperature=300, taut=5 * units.fs, trajectory='md.traj', logfile='md.log')
dyn.run(1000)'''
print('done')
