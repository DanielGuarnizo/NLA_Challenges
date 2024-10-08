%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## How to run the project 
- Execute the following comand in the terminal `./run.sh main ./data/images/256px-Albert_Einstein_Head.jpg`

## Note
- Task 8 works because previously is was use LIS library to generate sol_x.txt file, and then saved the result into MTX_objects direcotory
- command use to generated the sol_x.txt file was `mpirun -n 4 ./test1 /home/jellyfish/shared-folder/Challenge_1_NLA/data/MTX_objects/my_A_2.mtx /home/jellyfish/shared-folder/Challenge_1_NLA/data/MTX_objects/my_w.mtx /home/jellyfish/shared-folder/Challenge_1_NLA/data/MTX_objects/sol_x.txt hist_x.txt -i bicgstab -tol 10.0e-9
`