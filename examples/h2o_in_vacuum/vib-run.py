import os

from pathlib import Path
from ase.io import read, write
from ase.units import *
from ase.calculators.espresso import Espresso, EspressoProfile
from IR_with_ESM import Infrared_with_ESM
from IR_with_ESM.utils.analysis import Analysis
from ase.thermochemistry import IdealGasThermo
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
atoms = read("relaxed.pwo", index=-1)

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

calc_kwargs = {
    "input_data": input_data,
    "profile": profile,
    "pseudopotentials": pseudopotentials,
    "kpts": kpts,
    "directory": directory_name,
    "outdir": outdir,
    "file_name": file_name,
}

make_calculation_directory(directory_name=directory_name, outdir=outdir)

# Initialize Infrared_with_ESM
ir_with_esm = Infrared_with_ESM(
    atoms,
    calc_obj=calc_obj,
    calc_kwargs=calc_kwargs,
    indices=vibrational_indices,
    nfree=2,
)
write("check.traj", ir_with_esm.iterimages())
ir_with_esm.run()
ir_with_esm.summary(log="infrared_with_ESM.txt")
ir_with_esm.write_mode()

# Initialize analysis
an = Analysis(ir_with_esm)
an.export_gif()
an.get_spectra_plot(save="spectra_H2O_with_ESM.png")

# Calculate thermodynamic quantities
eq_atoms = read("output/eq/espresso.pwo")
potentialenergy = eq_atoms.get_potential_energy()
vib_energies = ir_with_esm.get_energies()
vib_frequencies = ir_with_esm.get_frequencies()
thermo = IdealGasThermo(
    vib_energies=vib_energies,
    potentialenergy=potentialenergy,
    atoms=eq_atoms,
    geometry="nonlinear",
    symmetrynumber=2,
    spin=0,
    ignore_imag_modes=True,
)
T_ambient = 298.15
P = 101325.0
H = thermo.get_enthalpy(temperature=T_ambient)
S = thermo.get_entropy(temperature=T_ambient, pressure=P)
G = thermo.get_gibbs_energy(temperature=T_ambient, pressure=P)
with open("helmholtz", "w") as helmholtz_file:
    helmholtz_file.write(f"Enthalpy         = {H:>20.11f} eV \n")
    helmholtz_file.write(f"Entropy          = {S:>20.11f} eV/K \n")
    helmholtz_file.write(f"TS               = {T_ambient * S:>20.11f} eV \n")
    helmholtz_file.write(f"Gibbs_energy     = {G:>20.11f} eV \n")
