import random
import numpy as np

random.seed(10)

# Number of jobs
n = 15
# Number of machines
m = 5
# Max time for random input generator
max_time = 100

# Generate processing times randomly
times = np.zeros((m, n))
tot_time_job = []
for i in range(m):
    for j in range(n):
        times[i][j] = random.randint(1, max_time)
# Total processing time per job - sum across machines
tot_processing_job = np.sum(times, axis=0)

def get_makespan(solution):
    ''' Calculate the makespan of a sequence of integer (jobs).
        - A job can start only after the previous operation of the same job in machine j-1 ends and
            machine is not processing any other job
        - Finish time of a job in a given machine is its start time plus processing time in current machine
    '''
    end_time = np.zeros((m, len(solution) + 1))
    for j in range(1, len(solution) + 1):
        end_time[0][j] = end_time[0][j - 1] + times[0][solution[j - 1]]
    for i in range(1, m):
        for j in range(1, len(solution) + 1):
            end_time[i][j] = max(end_time[i - 1][j], end_time[i][j - 1]) + times[i][solution[j - 1]]
    return end_time

def neh():
    ''' Heuristic NEH (Nawaz, Enscore & Ham) for flow shop scheduling
        1 - Start from an empty schedule
        2 - Add first the job with highest sum of processing time
        3 - Go through the list of the unassigned jobs, test all in all possible positions in the current solutions
        4 - Assign the best job at the best position (with lowest makespan) at the final solution
        5 - Repeat (3) and (4) until there are no job unassigned
    '''
    initial_solution = np.argsort(-tot_processing_job)
    current_solution = [initial_solution[0]]
    for i in range(1, n):
        best_cmax = 99999999
        for j in range(0, i + 1):
            temp_solution = current_solution[:]
            temp_solution.insert(j, initial_solution[i])
            temp_cmax = get_makespan(temp_solution)[m - 1][len(temp_solution)]
            if best_cmax > temp_cmax:
                best_seq = temp_solution
                best_cmax = temp_cmax
        current_solution = best_seq
    return current_solution, get_makespan(current_solution)[m - 1][n]

# True is warm start is used. False otherwise
use_warm_start = True

sequence_heuristic, cmax_heuristic = neh()

# Upper bound of the solution - sum of transit matrix
M = sum(times[j][i] for i in range(n) for j in range(m))

import gurobipy as grb

opt_model = grb.Model(name="Flow shop scheduling")

# Start time of job j at in machine i
x = {(j, i): opt_model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, name="x_{0}_{1}".format(j, i))
     for j in range(n) for i in range(m)}

# 1 if job j is executed before job k. 0 otherwise
y = {(j, k): opt_model.addVar(vtype=grb.GRB.BINARY, name="y_{0}_{1}".format(j, k))
     for j in range(n) for k in range(n) if j != k}

# Makespan - Completion time of last job in last machine
c = opt_model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, name="c")

# Job j in machine i can start only when it is finished in machine i-1
c1 = {(j, i): opt_model.addConstr(x[j, i] - x[j, i - 1] >= times[i - 1][j],
                                  name="c1_{0}_{1}".format(j, i))
      for j in range(n) for i in range(1, m)}


# Disjunctive constraints - if job j is after k, it should start after its completion
c2 = {(j, k, i): opt_model.addConstr(x[j, i] - x[k, i] + M * y[j, k] >= times[i][k],
                                     name="c2_{0}_{1}_{2}".format(j, k, i))
      for j in range(n) for k in range(n) for i in range(m) if k != j}

c3 = {(j, k, i): opt_model.addConstr(-x[j, i] + x[k, i] - M * y[j, k] >= times[i][j] - M,
                                     name="c3_{0}_{1}_{2}".format(j, k, i))
      for j in range(n) for k in range(n) for i in range(m) if k != j}

# Makespan is the completion time of last job in last machine
c4 = {j: opt_model.addConstr(c >= x[j, m - 1] + times[m - 1][j],
                             name="c4_{0}".format(j))
      for j in range(n)}

if use_warm_start:
    for j in range(n):
        for k in range(n):
            if j != k:
                y[j, k].Start = 0
    for i in range(0, len(sequence_heuristic) - 1):
        for j in range(i + 1, len(sequence_heuristic)):
            j1 = sequence_heuristic[i]
            j2 = sequence_heuristic[j]
            y[j1, j2].Start = 1
    c.Start = cmax_heuristic

# for minimization
opt_model.ModelSense = grb.GRB.MINIMIZE
opt_model.setObjective(c)
opt_model.setParam('MIPGap', 0.018)
opt_model.optimize()

