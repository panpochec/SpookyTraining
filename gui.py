from ase.io.trajectory import Trajectory
from ase.io import write
import ase.gui.gui


traj = Trajectory(filename='md.traj')
last = traj[0]
last = last
write('lastpoint.xyz', images=last)

