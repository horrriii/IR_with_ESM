#!/bin/bash
#PJM -L rscgrp=regular-o
#PJM -L node=10
#PJM --mpi proc=240
#PJM --omp thread=2
#PJM -L elapse=48:00:00
#PJM -g gf93
#PJM -o log
#PJM -N test-COads_1M-KOH
#PJM -e err
#PJM -o out

module purge
module load fjmpi/1.2.39 fj/1.2.39 odyssey fftw/3.3.9 mpi-fftw/3.3.9

echo "========= Job started  at `date` =========="

python3 vib-run.py

echo "========= Job finished at `date` =========="
