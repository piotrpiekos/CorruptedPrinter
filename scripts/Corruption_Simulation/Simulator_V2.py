# Author: Renzo Caballero
# KAUST: King Abdullah University of Science and Technology
# email: renzo.caballerorosas@kaust.edu.sa caballerorenzo@hotmail.com
# Website: renzocaballero.org, https://github.com/RenzoCab
# September 2023; Last revision: 15/09/2023

from scipy.io import loadmat
import csv
import numpy as np

corr_x    = [];
corr_y    = [];
x_ideal_t = [];
x_corr_t  = [];
u_x       = [];
y_ideal_t = [];
y_corr_t  = [];
u_y       = [];
num_lay   = [];

# Load the MATLAB file
data = loadmat('add_dudw_data.mat');
with open('./corrupted_pulleys/100029_2.csv', 'r') as csv_file:
#with open('./corrupted_pulleys/100032_1.csv', 'r') as csv_file:
#with open('./corrupted_pulleys/100336_1.csv', 'r') as csv_file:
#with open('./corrupted_pulleys/100343_2.csv', 'r') as csv_file:
#with open('./corrupted_pulleys/100344_1.csv', 'r') as csv_file:
#with open('./corrupted_pulleys/100349_2.csv', 'r') as csv_file:
#with open('./corrupted_pulleys/100471_1.csv', 'r') as csv_file:
#with open('./corrupted_pulleys/100606_1.csv', 'r') as csv_file:
#with open('./corrupted_pulleys/100730_1.csv', 'r') as csv_file:
#with open('./corrupted_pulleys/100730_2.csv', 'r') as csv_file:
#with open('./corrupted_pulleys/100827_1.csv', 'r') as csv_file:
#with open('./corrupted_pulleys/100887_1.csv', 'r') as csv_file:
#with open('./corrupted_pulleys/100887_2.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        corr_x.append(int(row[1]))
        corr_y.append(int(row[2]))
        num_lay.append(int(row[3]))
        x_ideal_t.append(int(row[4]))
        x_corr_t.append(float(row[5]))
        u_x.append(int(row[6]))
        y_ideal_t.append(int(row[9]))
        y_corr_t.append(float(row[10]))
        u_y.append(int(row[11]))
        

# Access MATLAB variables as Python dictionaries
array_corruptions = data['all_data']
array_corruptions = array_corruptions[0:5000];
print('The number of corruptions is',len(array_corruptions))
print('The length of each corruption is',len(array_corruptions[0]))

# x should be between 0 and 1100 (here we just choose 500 as an example):
x_ideal = 500;
x_corr  = 500;
# c (corruption) should be between 0 and 4999 (here we just choose 600 as an example):
c = array_corruptions[600];
# u (the control) should be either 0, 1, or -1 (here we just choose 1):
u = 1;
x_ideal_next = x_ideal + u;
x_corr_next  = x_corr  + u*c[x_ideal];
print('Then we move ideally from',x_ideal,'to',x_ideal_next)
print('Then we move corrupted from',x_corr,'to',x_corr_next)

num_layers = len(np.unique(num_lay))
elem_layer = np.unique(num_lay)

corruption_x = array_corruptions[int(corr_x[0])-1];
corruption_y = array_corruptions[int(corr_y[0])-1];

inc = 0;
accum_x = 0;
accum_y = 0;

for i in range(num_layers):
    
    for j in range(len(num_lay)):
        index = num_lay.index(elem_layer[i])

    we_start_in = index
    simulated_x_ideal = [x_ideal_t[we_start_in]];
    simulated_x_corr  = [x_corr_t[we_start_in]];
    simulated_y_ideal = [y_ideal_t[we_start_in]];
    simulated_y_corr  = [y_corr_t[we_start_in]];
    
    k = we_start_in;
    
    for j in range(num_lay.count(elem_layer[i])):
        
        simulated_x_ideal.append(simulated_x_ideal[j]+u_x[k+j])
        simulated_x_corr.append(simulated_x_corr[j]+u_x[k+j]*corruption_x[simulated_x_ideal[j]-1]);
        simulated_y_ideal.append(simulated_y_ideal[j]+u_y[k+j])
        simulated_y_corr.append(simulated_y_corr[j]+u_y[k+j]*corruption_y[simulated_y_ideal[j]-1]);
        accum_x = accum_x + abs(simulated_x_corr[j]-x_corr_t[k+j]);
        accum_y = accum_y + abs(simulated_y_corr[j]-y_corr_t[k+j]);
        inc = inc + 1;
        print('X_',inc,'_',accum_x);
        print('Y_',inc,'_',accum_y);
