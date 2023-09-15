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
loose_x   = [];
loose_y   = [];

# Load the MATLAB file
data = loadmat('add_dudw_data.mat');
#with open('./pulleys_and_belt/100028_1.csv', 'r') as csv_file:
#with open('./pulleys_and_belt/100029_1.csv', 'r') as csv_file:
#with open('./pulleys_and_belt/100031_1.csv', 'r') as csv_file:
#with open('./pulleys_and_belt/100032_2.csv', 'r') as csv_file:
#with open('./pulleys_and_belt/100343_1.csv', 'r') as csv_file:
#with open('./pulleys_and_belt/100388_1.csv', 'r') as csv_file:
#with open('./pulleys_and_belt/100638_1.csv', 'r') as csv_file:
#with open('./pulleys_and_belt/100638_2.csv', 'r') as csv_file:
#with open('./pulleys_and_belt/100641_1.csv', 'r') as csv_file:
#with open('./pulleys_and_belt/100827_2.csv', 'r') as csv_file:
#with open('./pulleys_and_belt/101168_1.csv', 'r') as csv_file:
with open('./pulleys_and_belt/103355_2.csv', 'r') as csv_file:
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
        loose_x.append(int(row[14]))
        loose_y.append(int(row[15]))
        
# Access MATLAB variables as Python dictionaries
array_corruptions = data['all_data']
array_corruptions = array_corruptions[0:5000];

num_layers = len(np.unique(num_lay))
elem_layer = np.unique(num_lay)

corruption_x = array_corruptions[int(corr_x[0])-1];
corruption_y = array_corruptions[int(corr_y[0])-1];

inc = 0;
accum_x = 0;
accum_y = 0;

loose_x = loose_x[0];
loose_y = loose_y[0];

for i in range(num_layers):
    
    for j in range(len(num_lay)):
        index = num_lay.index(elem_layer[i])

    we_start_in = index
    simulated_x_ideal = [x_ideal_t[we_start_in]];
    simulated_x_corr  = [x_corr_t[we_start_in]];
    simulated_y_ideal = [y_ideal_t[we_start_in]];
    simulated_y_corr  = [y_corr_t[we_start_in]];
    
    k = we_start_in;
    
    EP_x = 0;
    EP_y = 0;
    EN_x = loose_x;
    EN_y = loose_y;
    if_x = loose_x;
    if_y = loose_y;
    
    x_motor = [x_ideal_t[we_start_in]];
    y_motor = [y_ideal_t[we_start_in]];
    
    for j in range(num_lay.count(elem_layer[i])):
        
        x_motor.append(x_motor[j]+u_x[k+j]);
        y_motor.append(y_motor[j]+u_y[k+j]);
        
        if u_x[k+j] == 1 and EP_x == 0:
            simulated_x_corr.append(simulated_x_corr[j]+u_x[k+j]*corruption_x[x_motor[j]-1]);
        elif u_x[k+j] == 1 and EP_x != 0:
            simulated_x_corr.append(simulated_x_corr[j]);
            reduction_x = EP_x - u_x[k+j]*corruption_x[x_motor[j]-1];
            EP_x = max(0,reduction_x);
            EN_x = min(if_x,EN_x + u_x[k+j]*corruption_x[x_motor[j]-1])
            if reduction_x < 0:
                simulated_x_corr[-1] = simulated_x_corr[-1] - reduction_x;
            
        if u_x[k+j] == -1 and EN_x == 0:
            simulated_x_corr.append(simulated_x_corr[j]+u_x[k+j]*corruption_x[x_motor[j]-1]);
        elif u_x[k+j] == -1 and EN_x != 0:
            simulated_x_corr.append(simulated_x_corr[j]);
            reduction_x = EN_x + u_x[k+j]*corruption_x[x_motor[j]-1];
            EN_x = max(0,reduction_x);
            EP_x = min(if_x,EP_x - u_x[k+j]*corruption_x[x_motor[j]-1])
            if reduction_x < 0:
                simulated_x_corr[-1] = simulated_x_corr[-1] - reduction_x;
            
        if u_x[k+j] == 0:
            simulated_x_corr.append(simulated_x_corr[j]);
            
        if u_y[k+j] == 1 and EP_y == 0:
            simulated_y_corr.append(simulated_y_corr[j]+u_y[k+j]*corruption_y[y_motor[j]-1]);
        elif u_y[k+j] == 1 and EP_y != 0:
            simulated_y_corr.append(simulated_y_corr[j]);
            reduction_y = EP_y - u_y[k+j]*corruption_y[y_motor[j]-1];
            EP_y = max(0,reduction_y);
            EN_y = min(if_y,EN_y + u_y[k+j]*corruption_y[y_motor[j]-1])
            if reduction_y < 0:
                simulated_y_corr[-1] = simulated_y_corr[-1] - reduction_y;
            
        if u_y[k+j] == -1 and EN_y == 0:
            simulated_y_corr.append(simulated_y_corr[j]+u_y[k+j]*corruption_y[y_motor[j]-1]);
        elif u_y[k+j] == -1 and EN_y != 0:
            simulated_y_corr.append(simulated_y_corr[j]);
            reduction_y = EN_y + u_y[k+j]*corruption_y[y_motor[j]-1];
            EN_y = max(0,reduction_y);
            EP_y = min(if_y,EP_y - u_y[k+j]*corruption_y[y_motor[j]-1])
            if reduction_y < 0:
                simulated_y_corr[-1] = simulated_y_corr[-1] - reduction_y;
            
        if u_y[k+j] == 0:
            simulated_y_corr.append(simulated_y_corr[j]);
        
        simulated_x_ideal.append(simulated_x_ideal[j]+u_x[k+j])
        simulated_y_ideal.append(simulated_y_ideal[j]+u_y[k+j])
        
        accum_x = accum_x + abs(simulated_x_corr[j]-x_corr_t[k+j]);
        accum_y = accum_y + abs(simulated_y_corr[j]-y_corr_t[k+j]);
        inc = inc + 1;
        print('X_',inc,'_',accum_x);
        print('Y_',inc,'_',accum_y);