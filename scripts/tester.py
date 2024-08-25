import os
import sys
from datetime import datetime
import pickle

# Import functions
from stencil2d_jax_base import calculations as jax_base_calc
from stencil2d_jax import calculations as jax_calc
from stencil2d_numpy import calculations as numpy_calc
from stencil2d_torch import calculations as torch_calc

# Define paramters to test
range_nx = [128]
range_ny = [128]
range_nz = [64]
range_num_iter = [128]
range_precision = ["32", "64"]
functions = [jax_base_calc, jax_calc, numpy_calc, torch_calc]

results = {}


def tester():
    for nx, ny, nz, num_iter in zip(range_nx, range_ny, range_nz, range_num_iter):
        for p in range_precision:
            for f in functions:
                results[f"nx{nx}_ny{ny}_nz{nz}_num_iter{num_iter}_p{p}_f{f}"] = f(
                    nx, ny, nz, num_iter, 2, p, return_result=False, return_time=True
                )

    with open(f"../results/{datetime.now().strftime('%Y%m%dT%H%M%S')}.pkl", "wb") as f_out:
        pickle.dump(results, f_out)


def main():
    tester()


# Run with something like: srun --account=classXXX --constraint=gpu --partition=normal --nodes=1 --ntasks-per-core=1 --ntasks-per-node=1 --cpus-per-task=12 --hint=nomultithread python3 tester.py

if __name__ == "__main__":
    os.chdir(sys.path[0])  # Change the directory
    main()
