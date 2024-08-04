import os
import sys

# Import functions
from stencil2d_jax_base import calculations as jax_base_calc
from stencil2d_jax import calculations as jax_calc
from stencil2d_numpy import calculations as numpy_calc
from stencil2d_torch_base import calculations as torch_calc

# Define paramters to test
range_nx = []
range_ny = []
range_nz = []
range_num_iter = []
range_precision = ["32", "64"]
devices = ["cpu", "gpu"]
functions = [jax_base_calc, jax_calc, numpy_calc, torch_calc]

RESULT_PATH = ""


def tester():
    for nx in range_nx:
        for ny in range_ny:
            for nz in range_nz:
                for num_iter in range_num_iter:
                    for p in range_precision:
                        for d in devices:
                            for f in functions:
                                result = f(nx, ny, nz, num_iter, RESULT_PATH, 2, p, return_result=True)
                                # TODO: print result to correct results folder
                                with open(RESULT_PATH, "w") as out_file:
                                    out_file.write(result)


def main():
    tester()


# Run with something like: srun --account=classXXX --constraint=gpu --partition=normal --nodes=1 --ntasks-per-core=1 --ntasks-per-node=1 --cpus-per-task=12 --hint=nomultithread python3 tester.py
# TODO: make functions callable with gpu / cpu
if __name__ == "__main__":
    os.chdir(sys.path[0])  # Change the directory
    main()
