# HW3 Code

Here is where I developed the code for homework three, which was exploring matrix-vector optimization methods. 
Code is compiled and run on a SLURM environment by running `sbatch run.sh <option> <output_file>`, where `<option>` can be 1, 2, 3, 4, 5, ... and indicates single-thread per row, single-warp per row, multiple-warp per row, or single-block for multiple rows.

All methods for experimental methods (i.e. ) are stored in `matrixVectorMultiplication.cu`, and I ran experiments by changing the `option` argument. 
I also have a function to generate the Latex tables seen in the report.
