import os

from pathlib import Path
from ase.io import read, write
from ase.units import *
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.vibrations import Infrared_with_ESM
from ase.utils.analysis import Analysis
from ase.wisteria import make_calculation_directory

ecutwfc = 850 * (eV / Ry)
tot_charge = 0.0
directory_name = "DFT"
outdir = "tmp"
file_name = "H2O_IR"
vibrational_indices = [0, 1, 2]

pseudopotentials = {
    "H": "H.pbe-kjpaw_psl.1.0.0.UPF",
    "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Cu": "Cu.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
}

kpts = (1, 1, 1)

input_data = {
    "control": {
        "calculation": "scf",
        "restart_mode": "from_scratch",
        "wf_collect": True,
        "nstep": 1,
        "verbosity": "high",
        "tprnfor": True,
        "tstress": False,
        "pseudo_dir": str(Path("~/QE/pseudo/").expanduser()),
    },
    "system": {
        "ecutwfc": ecutwfc,
        "ecutrho": ecutwfc * 10,
        "nosym": False,
        "input_dft": "vdW-DF2",
        "tot_charge": tot_charge,
        "assume_isolated": "esm",
        "esm_bc": "bc1",
        "nbnd": 40,
    },
    "electrons": {
        "mixing_beta": 1 / 3,
        "mixing_mode": "plain",
        "mixing_ndim": 12,
        "diagonalization": "david",
        "electron_maxstep": 800,
        "conv_thr": 1.0e-10 * (eV / Ry),
    },
    "ions": {
        "ion_dynamics": "bfgs",
    },
}

atoms = read("relaxed.pwo",index=-1)

make_calculation_directory(directory_name=directory_name, outdir=outdir)

calc_obj = Espresso
profile = EspressoProfile(
    argv=[
        "mpirun",
        "-np",
        os.environ.get("PJM_MPI_PROC"),
        "--stdout",
        "espresso.pwo",
        "/work/gf93/share/espresso/qe-rism-tuned/bin/pw.x",
    ]
)

calc_kwargs = {
    "input_data": input_data,
    "profile": profile,
    "pseudopotentials": pseudopotentials,
    "kpts": kpts,
    "directory": directory_name,
    "outdir": outdir,
    "file_name": file_name,
}

ir_with_esm = Infrared_with_ESM(
    atoms, calc_obj=calc_obj, calc_kwargs=calc_kwargs, indices=vibrational_indices, nfree=2
)
write("check.traj", ir_with_esm.iterimages())
ir_with_esm.run()
ir_with_esm.summary(log="infrared_with_ESM.txt")
ir_with_esm.write_mode()
an = Analysis(ir_with_esm)
an.export_gif()
an.get_spectra_plot(save="spectra_H2O_with_ESM.png")