## TASK 3: 
- Command use to compute greater eigen value of $A^{T}*A$
```
mpirun -n 4 ./eigen1 /home/jellyfish/shared-folder/NLA_Challenges/Challenge_2/data/MTX_onjects/A_transpose_A.mtx /home/jellyfish/shared-folder/NLA_Challenges/Challenge_2/data/MTX_onjects/eigvec.txt hist.txt -e pi -etol 10.e-8

```

- Here are the results 
```
number of processes = 4
matrix size = 256 x 256 (65536 nonzero entries)

initial vector x      : all components set to 1
precision             : double
eigensolver           : Power
convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-07 * ||lx||_2
matrix storage format : CSR
shift                 : 0.000000e+00
eigensolver status    : normal end

Power: mode number          = 0
Power: eigenvalue           = 1.608332e+04
Power: number of iterations = 7
Power: elapsed time         = 7.745830e-04 sec.
Power:   preconditioner     = 0.000000e+00 sec.
Power:     matrix creation  = 0.000000e+00 sec.
Power:   linear solver      = 0.000000e+00 sec.
Power: relative residual    = 2.151877e-08


```

- The Eigenvalue computed corresponds to the singular value $\sqrt{1.608332e+04} = 126.82$
  
## TASK 4: 
- Command with the shift (5.946860e+02) included.
```
mpirun -n 4 ./eigen1 /home/jellyfish/shared-folder/NLA_Challenges/Challenge_2/data/MTX_onjects/A_transpose_A.mtx /home/jellyfish/shared-folder/NLA_Challenges/Challenge_2/data/MTX_onjects/eigvec.txt hist.txt -e pi -etol 10.e-8 -shift 5.946860e+02

```
- New number of iterations needed to reach convergence = 6
