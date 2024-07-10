# Testing
## Instructions
To test whether your implementation is correct (with respect to the baseline of the course), you first need to save the result of your version into a file. You can use and probably should use `np.save()` such that the results are consistent and for that matter it probably makes sense to convert the output of your implementation in a specific data model to a `numpy` array. You should name the file as follows `f"{path}/{datetime.now().strftime("%Y%m%dT%H%M%S")}-nx{nx}_ny{ny}_nz{nz}_iter{num_iter}_halo{num_halo}.npy"` such that the solution checker knows which solution to compare your output to and generate the necessary one if it does not exist yet. Once you have the file with your output, you can run the following script as follows to check whether it gives the correct result:
```bash
python3 scripts/check_solution.py -s <path-to-data>
```

You should always specify the paths relative to the location of the script and not from where you are runnign the script.
