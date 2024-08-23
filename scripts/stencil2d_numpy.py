import os
import sys
import click
import numpy as np
import time
from datetime import datetime


def laplacian(in_field, lap_field, num_halo, extend=0):
    ib = num_halo - extend
    ie = -num_halo + extend
    jb = num_halo - extend
    je = -num_halo + extend

    lap_field[:, jb:je, ib:ie] = (
        -4.0 * in_field[:, jb:je, ib:ie]
        + in_field[:, jb:je, ib - 1 : ie - 1]
        + in_field[:, jb:je, ib + 1 : ie + 1 if ie != -1 else None]
        + in_field[:, jb - 1 : je - 1, ib:ie]
        + in_field[:, jb + 1 : je + 1 if je != -1 else None, ib:ie]
    )


def update_halo(field, num_halo):
    field[:, :num_halo, num_halo:-num_halo] = field[:, -2 * num_halo : -num_halo, num_halo:-num_halo]
    field[:, -num_halo:, num_halo:-num_halo] = field[:, num_halo : 2 * num_halo, num_halo:-num_halo]
    field[:, :, :num_halo] = field[:, :, -2 * num_halo : -num_halo]
    field[:, :, -num_halo:] = field[:, :, num_halo : 2 * num_halo]


def apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=1):
    tmp_field = np.empty_like(in_field)
    alpha_neg = -alpha

    for n in range(num_iter):
        update_halo(in_field, num_halo)

        laplacian(in_field, tmp_field, num_halo=num_halo, extend=1)
        laplacian(tmp_field, out_field, num_halo=num_halo, extend=0)

        in_core = in_field[:, num_halo:-num_halo, num_halo:-num_halo]
        out_core = out_field[:, num_halo:-num_halo, num_halo:-num_halo]

        np.add(in_core, alpha_neg * out_core, out=out_core)

        if n < num_iter - 1:
            in_field, out_field = out_field, in_field
        else:
            update_halo(out_field, num_halo)

        if num_iter % 2 == 0:
            in_field, out_field = out_field, in_field

    return out_field


def calculations(nx, ny, nz, num_iter, num_halo, precision, result_dir="", return_result=False, return_time=False):
    assert 0 < nx <= 1024 * 1024, "You have to specify a reasonable value for nx"
    assert 0 < ny <= 1024 * 1024, "You have to specify a reasonable value for ny"
    assert 0 < nz <= 1024, "You have to specify a reasonable value for nz"
    assert 0 < num_iter <= 1024 * 1024, "You have to specify a reasonable value for num_iter"
    assert 2 <= num_halo <= 256, "You have to specify a reasonable number of halo points"
    alpha = 1.0 / 32.0

    dtype = np.float64 if precision == "64" else np.float32

    in_field = np.zeros((nz, ny + 2 * num_halo, nx + 2 * num_halo), dtype=dtype)
    in_field[
        nz // 4 : 3 * nz // 4,
        num_halo + ny // 4 : num_halo + 3 * ny // 4,
        num_halo + nx // 4 : num_halo + 3 * nx // 4,
    ] = 1.0
    out_field = np.copy(in_field)

    apply_diffusion(in_field, out_field, alpha, num_halo)

    tic = time.time()
    out_field = apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=num_iter)
    toc = time.time()

    print(f"Elapsed time for work = {toc - tic} s")

    if result_dir != "":
        result_path = f"{result_dir}/{datetime.now ().strftime ('%Y%m%dT%H%M%S')}-nx{nx}_ny{ny}_nz{nz}_iter{num_iter}_halo{num_halo}_p{precision}.npy"
        np.save(result_path, out_field)

    if return_time and return_result:
        return out_field, toc - tic

    if return_time:
        return toc - tic

    if return_result:
        return out_field


@click.command()
@click.option("--nx", type=int, required=True, help="Number of gridpoints in x-direction")
@click.option("--ny", type=int, required=True, help="Number of gridpoints in y-direction")
@click.option("--nz", type=int, required=True, help="Number of gridpoints in z-direction")
@click.option("--num_iter", type=int, required=True, help="Number of iterations")
@click.option("--precision", type=click.Choice(["64", "32"]), default="64", required=True, help="Precision")
@click.option(
    "--num_halo",
    type=int,
    default=2,
    help="Number of halo-pointers in x- and y-direction",
)
@click.option(
    "--result_dir",
    type=str,
    default="../data/numpy",
    help="Specify the folder where the results should be saved (relative to the location of the script or absolute).",
)
def main(nx, ny, nz, num_iter, result_dir, num_halo, precision):
    calculations(nx, ny, nz, num_iter, num_halo, precision, result_dir=result_dir, return_result=False)


if __name__ == "__main__":
    os.chdir(sys.path[0])  # Change the directory
    main()
