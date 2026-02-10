import os

from pathlib import Path
from ase.io import read, write
from ase.units import *
from ase.constraints import FixAtoms
from ase.calculators.espresso import Espresso, EspressoProfile
from IR_with_ESM import Infrared_with_ESM
from IR_with_ESM.utils.analysis import Analysis
from ase.thermochemistry import HarmonicThermo
from ase.wisteria import make_calculation_directory

# Set the calculator object
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

# Set calculator object kwargs
slab_structure = read("slab-relax.traj", index=-1)

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

constrain_layers = FixAtoms(indices=[0, 1, 2, 3, 4, 5, 6, 7, 8])
slab_structure.set_constraint(constrain_layers)

solute_ljs = []
solute_epsilons = []
solute_sigmas = []
for i in slab_structure.numbers:
    if i == 29:
        solute_ljs.append("none")
        solute_epsilons.append(1.0)
        solute_sigmas.append(3.0)
    elif i == 6:
        solute_ljs.append("opls-aa")
        solute_epsilons.append(None)
        solute_sigmas.append(None)
    else:
        solute_ljs.append("opls-aa")
        solute_epsilons.append(None)
        solute_sigmas.append(None)
slab_structure.set_solute_lj(solute_ljs=solute_ljs)
slab_structure.set_solute_epsilon(solute_epsilons=solute_epsilons)
slab_structure.set_solute_sigma(solute_sigmas=solute_sigmas)

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

make_calculation_directory(directory_name=directory_name, outdir=outdir)

# Initialize Infrared_with_ESM
ir_with_esm = Infrared_with_ESM(
    slab_structure,
    calc_obj=calc_obj,
    calc_kwargs=calc_kwargs,
    indices=vibrational_indices,
    nfree=2,
)
write("check.traj", ir_with_esm.iterimages())
ir_with_esm.run()
ir_with_esm.summary(log="infrared_with_ESM-RISM.txt")
ir_with_esm.write_mode()

# Initialize analysis
an = Analysis(ir_with_esm)
an.export_gif()
an.get_spectra_plot(save="spectra_COads_with_ESM-RISM.png")

# Calculate thermodynamic quantities
eq_atoms = read('output/eq/espresso.pwo')
potentialenergy = eq_atoms.get_potential_energy()
vib_energies = ir_with_esm.get_energies()
vib_frequencies = ir_with_esm.get_frequencies()
thermo = HarmonicThermo(
    vib_energies=vib_energies, potentialenergy=potentialenergy, ignore_imag_modes=True
)
T_ambient = 298.15
U = thermo.get_internal_energy(temperature=T_ambient)
S = thermo.get_entropy(temperature=T_ambient)
A = thermo.get_helmholtz_energy(temperature=T_ambient)
with open("helmholtz", "w") as helmholtz_file:
    helmholtz_file.write(f"Internal_energy  = {U:>20.11f} eV \n")
    helmholtz_file.write(f"Entropy          = {S:>20.11f} ev/K \n")
    helmholtz_file.write(f"TS               = {T_ambient * S:>20.11f} eV \n")
    helmholtz_file.write(f"Helmholtz_energy = {A:>20.11f} eV \n")