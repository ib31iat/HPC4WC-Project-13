import os
import sys

# Import functions
from stencil2d_jax_base import calculations as jax_base_calc
from stencil2d_jax import calculations as jax_calc
from stencil2d_numpy import calculations as numpy_calc
from stencil2d_torch import calculations as torch_calc

# Define paramters to test
range_nx = []
range_ny = []
range_nz = []
range_num_iter = []
range_precision = ["32", "64"]
devices = ["cpu", "gpu"]
functions = [jax_base_calc, jax_calc, numpy_calc, torch_calc]


def tester():
    for nx, ny, nz, num_iter in zip(range_nx, range_ny, range_nz, range_num_iter):
        for p in range_precision:
            for d in devices:
                for f in functions:
                    print(f"nx{nx}_ny{ny}_nz{nz}_num_iter{num_iter}_p{p}_d{d}_f{f}")
                    f(nx, ny, nz, num_iter, "", 2, p, return_result=False, use_gpu=(d == "gpu"))


def main():
    tester()


# Run with something like: srun --account=classXXX --constraint=gpu --partition=normal --nodes=1 --ntasks-per-core=1 --ntasks-per-node=1 --cpus-per-task=12 --hint=nomultithread python3 tester.py

if __name__ == "__main__":
    os.chdir(sys.path[0])  # Change the directory
    main()
