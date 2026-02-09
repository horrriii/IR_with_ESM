import os

from pathlib import Path
from ase.io import read, write
from ase.units import *
from ase.constraints import FixAtoms
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.vibrations import Infrared_with_ESM
from ase.utils.analysis import Analysis
from ase.wisteria import make_calculation_directory

ecutwfc = 850 * (eV / Ry)
tot_charge = 0.0
directory_name = "DFT"
outdir = "tmp"
file_name = "Cu-vib"
vibrational_indices = [-1, -2]

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

kpts = (8, 8, 1)

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
        "occupations": "smearing",
        "smearing": "marzari-vanderbilt",
        "degauss": 0.1 * (eV / Ry),
        "nosym": True,
        "input_dft": "vdW-DF2",
        "tot_charge": tot_charge,
        "assume_isolated": "esm",
        "esm_bc": "bc1",
    },
    "electrons": {
        "mixing_beta": 1 / 3,
        "mixing_mode": "plain",
        "mixing_ndim": 12,
        "diagonalization": "rmm",
        "diago_rmm_conv": False,
        "diago_rmm_ndim": 8,
        "electron_maxstep": 2000,
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
            "starting1d": "file",
            "rism1d_conv_thr": 1.0e-8,
            "rism1d_maxstep": 100000,
            "mdiis1d_size": 20,
            "mdiis1d_step": 0.1,
            "starting3d": "file",
            "rism3d_conv_thr": 1.0e-6,
            "rism3d_conv_level": 0.8,
            "mdiis3d_size": 20,
            "mdiis3d_step": 0.8,
            "rism3d_maxstep": 10000,
            "laue_expand_right": 60.0,
            "laue_starting_right": -8.20,
            "laue_buffer_right": 10.0,
    },
}

slab_structure = read("slab-relax.traj", index=-1)

constrain_layers = FixAtoms(indices=[0, 1, 2, 3, 4, 5, 6, 7, 8])
slab_structure.set_constraint(constrain_layers)

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
    'solvents_info':solvents_info,
}

ir_with_esm = Infrared_with_ESM(
    slab_structure, calc_obj=calc_obj, calc_kwargs=calc_kwargs, indices=vibrational_indices, nfree=2
)
write("check.traj", ir_with_esm.iterimages())
ir_with_esm.run()
ir_with_esm.summary(log="infrared_with_ESM-RISM.txt")
ir_with_esm.write_mode()

an = Analysis(ir_with_esm)
an.export_gif()
an.get_spectra_plot(save="spectra_COads_with_ESM-RISM.png")