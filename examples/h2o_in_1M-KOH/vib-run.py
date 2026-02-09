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
file_name = "H2O-IR_ESM-RISM"
vibrational_indices = [0, 1, 2]

pseudopotentials = {
    "H": "H.pbe-kjpaw_psl.1.0.0.UPF",
    "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Cu": "Cu.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
}

solvents_info = {
    "density_unit": "mol/L",
    "H2O": [-1.0, "H2O.tip5p.MOL"],
    "K+": [1.0, "K+.oplsaa.MOL"],
    "OH-": [1.0, "OH-.oplsaa.MOL"],
}

kpts = (1, 1, 1)

input_data = {
    "control": {
        "calculation": "scf",
        "restart_mode": "from_scratch",
        "wf_collect": True,
        "nstep": 1,
        "title": file_name,
        "verbosity": "high",
        "outdir": outdir,
        "prefix": file_name,
        "tprnfor": True,
        "tstress": False,
        "trism": True,
        "pseudo_dir": str(Path("~/QE/pseudo/").expanduser()),
    },
    "system": {
        "ecutwfc": ecutwfc,
        "ecutrho": ecutwfc * 10,
        # "occupations": "smearing",
        # "smearing": "marzari-vanderbilt",
        # "degauss": 0.1 * (eV / Ry),
        "tot_charge": tot_charge,
        "input_dft": "vdW-DF2",
        "assume_isolated": "esm",
        "esm_bc": "bc1",
        "nbnd": 20,
    },
    "electrons": {
        "mixing_beta": 1 / 3,
        "mixing_mode": "plain",
        "mixing_ndim": 12,
        "diagonalization": "rmm",
        "diago_rmm_conv": False,
        "diago_rmm_ndim": 4,
        "electron_maxstep": 800,
        "conv_thr": 1.0e-10 * (eV / Ry),
    },
    "ions": {
        "ion_dynamics": "bfgs",
    },
    "rism": {
        "nsolv": 3,
        "closure": "kh",
        "tempv": 300.00,
        "ecutsolv": ecutwfc * 4,
        "laue_expand_right": 60.0,
        "laue_expand_left": 60.0,
        "starting1d": "zero",
        "rism1d_conv_thr": 1.0e-8,
        "rism1d_maxstep": 100000,
        "mdiis1d_size": 20,
        "mdiis1d_step": 0.1,
        "starting3d": "zero",
        "rism3d_conv_thr": 1.0e-6,
        "rism3d_conv_level": 0.5,
        "mdiis3d_size": 20,
        "mdiis3d_step": 0.8,
        "rism3d_maxstep": 10000,
    },
}

atoms = read("relaxed.pwo", index=-1)

solute_ljs = []
solute_epsilons = []
solute_sigmas = []
for i in atoms.numbers:
    if i == 29:
        solute_ljs.append("none")
        solute_epsilons.append(1.0)
        solute_sigmas.append(3.0)
    elif i == 1:
        solute_ljs.append("none")
        solute_epsilons.append(0.046)
        solute_sigmas.append(1.00)
    else:
        solute_ljs.append("none")
        solute_epsilons.append(0.1554)
        solute_sigmas.append(3.1660)
atoms.set_solute_lj(solute_ljs=solute_ljs)
atoms.set_solute_epsilon(solute_epsilons=solute_epsilons)
atoms.set_solute_sigma(solute_sigmas=solute_sigmas)

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
    "solvents_info": solvents_info,
}

ir_with_esm = Infrared_with_ESM(
    atoms,
    calc_obj=calc_obj,
    calc_kwargs=calc_kwargs,
    indices=vibrational_indices,
    nfree=2,
)
write("check.traj", ir_with_esm.iterimages())
ir_with_esm.run()
ir_with_esm.summary(log="infrared_with_ESM-RISM.txt")
ir_with_esm.write_mode()
an = Analysis(ir_with_esm)
an.export_gif()
an.get_spectra_plot(save="spectra_H2O_with_ESM-RISM.png")