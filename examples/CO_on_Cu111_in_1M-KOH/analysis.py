from ase.io import read
from ase.units import *
from ase.vibrations import Infrared_with_ESM
from ase.utils.analysis import Analysis
from ase.calculators.emt import EMT

pseudopotentials = {
    "H": "H.pbe-kjpaw_psl.1.0.0.UPF",
    "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Cu": "Cu.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
}

atoms = read("./../relax/slab-relax.traj", index=-1)

calc_obj = EMT
ir_with_esm = Infrared_with_ESM(atoms, calc_obj=calc_obj, indices=[-1, -2], nfree=2)

an = Analysis(ir_with_esm)
an.export_gif()
an.get_spectra_plot(save="spectra_COads_with_ESM-RISM.png")
