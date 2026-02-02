import os
# import psutil
import sys
from time import time
from pysat.solvers import Solver
from pysat.card import ITotalizer



def create_var_map(num_var, num_frequencies):
    var_map = [[0] * (num_var + 1) for _ in range(num_frequencies + 1)]

    top = 0
    for i in range(1, num_var + 1):
        for j in range(1, num_frequencies + 1):
            top += 1
            var_map[j][i] = top 

def create_penalty_var_map(last_var_num,):
    penalty_var_map = {}
    

    

    return 

    
    
                            
def create_label_var_map(labels, start_index):
    label_var_map = {}
    current = start_index
    for lb in labels:
        label_var_map[lb] = current
        current += 1
    return label_var_map
    
# ánh xạ biến active -> biến xác nhận label được sử dụng    
def build_label_constraints(solver, var_map, label_var_map):
    for (i, v), varnum in var_map.items():
        lb_varnum = label_var_map[v]
        solver.add_clause([-varnum, lb_varnum])

def amk_nsc(solver, lits, K):

    if isinstance(lits, dict):
        lits = list(lits.values())

    n = len(lits)
    top = solver.nof_vars()

    # r[i][j] với i = 1..n, j = 1..K
    r = [[0] * (K + 1) for _ in range(n + 1)]

    for i in range(1, K):
        for j in range(1, i + 1):
            top += 1
            r[i][j] = top
    for i in range(K, n + 1):
        for j in range(1, K + 1):
            top += 1
            r[i][j] = top

    # (1)  ¬x_i ∨ r(i,1)
    for i in range(1, n + 1):
        solver.add_clause([-lits[i - 1], r[i][1]])

    # (2)  ¬r(i-1,j) ∨ r(i,j)
    for i in range(2, n + 1):
        for j in range(1, min(i - 1, K) + 1):
            solver.add_clause([-r[i - 1][j], r[i][j]])

    # (3)  ¬x_i ∨ ¬r(i-1,j-1) ∨ r(i,j)
    for i in range(2, n + 1):
        for j in range(2, min(i, K) + 1):
            solver.add_clause([-lits[i - 1], -r[i - 1][j - 1], r[i][j]])

    # (5)  x_i ∨ ¬r(i,i)
    for i in range(1, K + 1):
        solver.add_clause([lits[i - 1], -r[i][i]])

    # (6)  r(i-1,j-1) ∨ ¬r(i,j)
    for i in range(2, n + 1):
        for j in range(2, min(i, K) + 1):
            solver.add_clause([r[i - 1][j - 1], -r[i][j]])

    # (7)  x_i ∨ r(i-1,j-1) ∨ ¬r(i,j)
    for i in range(2, n + 1):
        for j in range(1, min(i - 1, K) + 1):
            solver.add_clause([lits[i - 1], r[i - 1][j], -r[i][j]])

    # (8)  ¬x_i ∨ ¬r(i-1,K)
    for i in range(K + 1, n + 1):
        solver.add_clause([-lits[i - 1], -r[i - 1][K]])

    # rhs[j-1] ⇔ sum(lits) ≤ j
    rhs = [r[n][j] for j in range(1, K + 1)]
    return rhs



def amk_nsc_reduced(solver, lits, K):
    if isinstance(lits, dict):
        lits = list(lits.values())

    n = len(lits)
    top = solver.nof_vars()

    # r[i][j] với i = 1..n, j = 1..K
    r = [[0] * (K + 1) for _ in range(n + 1)]

    # tạo biến phụ
    for i in range(1, K):
        for j in range(1, i + 1):
            top += 1
            r[i][j] = top
    for i in range(K, n + 1):
        for j in range(1, K + 1):
            top += 1
            r[i][j] = top

    # (1)  ¬x_i ∨ r(i,1)
    for i in range(1, n + 1):
        solver.add_clause([-lits[i - 1], r[i][1]])

    # (2)  ¬r(i-1,j) ∨ r(i,j)
    for i in range(2, n + 1):
        for j in range(1, min(i - 1, K) + 1):
            solver.add_clause([-r[i - 1][j], r[i][j]])

    # (3)  ¬x_i ∨ ¬r(i-1,j-1) ∨ r(i,j)
    for i in range(2, n + 1):
        for j in range(2, min(i, K) + 1):
            solver.add_clause([-lits[i - 1], -r[i - 1][j - 1], r[i][j]])

    # (8)  ¬x_i ∨ ¬r(i-1,K)
    for i in range(K + 1, n + 1):
        solver.add_clause([-lits[i - 1], -r[i - 1][K]])

    # rhs[j-1] ⇔ sum(lits) ≤ j
    rhs = [r[n][j] for j in range(1, K + 1)]
    return rhs

def amk_sc(solver, lits, K):
    if isinstance(lits, dict):
        lits = list(lits.values())
    n = len(lits)
    top = solver.nof_vars()

    # s[i][j] : i in [0..n-1], j in [0..K-1]
    s = [[0] * (K + 1) for _ in range(n + 1)]

    # tạo biến phụ
    for i in range(1, n + 1):
        for j in range(1, K + 1):
            top += 1
            s[i][j] = top    

    solver.add_clause([-lits[0], s[1][1]])  # (1)   
    for j in range(2, K + 1):
        solver.add_clause([-s[1][j]])       # (2)

    for i in range(2, n + 1):
        solver.add_clause([-lits[i - 1], s[i][1]])  # (3)
        solver.add_clause([-s[i - 1][1], s[i][1]])      # (4)
        for j in range(2, K + 1):
            solver.add_clause([-lits[i - 1], -s[i - 1][j - 1], s[i][j]])  # (5)
            solver.add_clause([-s[i - 1][j], s[i][j]])  # (6)
        solver.add_clause([-lits[i - 1], -s[i - 1][K]])  # (7)
    

    rhs = [s[n][j] for j in range(1, K + 1)]

    return rhs

def amk_sc_reduced(solver, lits, K):
    if isinstance(lits, dict):
        lits = list(lits.values())
    
    n = len(lits)
    top = solver.nof_vars()

    # s[i][j] : i in [0..n-1], j in [0..K-1]
    s = [[0] * K for _ in range(n)]

    # tạo biến phụ
    for i in range(n):
        for j in range(K):
            top += 1
            s[i][j] = top

    # (1) ¬x_i ∨ s[i][0]
    for i in range(n):
        solver.add_clause([-lits[i], s[i][0]])

    # (2) ¬s[i-1][j] ∨ s[i][j]
    for i in range(1, n):
        for j in range(K):
            solver.add_clause([-s[i-1][j], s[i][j]])

    # (3) ¬x_i ∨ ¬s[i-1][j-1] ∨ s[i][j]
    for i in range(1, n):
        for j in range(1, K):
            solver.add_clause([-lits[i], -s[i-1][j-1], s[i][j]])

    # (4) ¬x_i ∨ ¬s[i-1][K-1]
    for i in range(1, n):
        solver.add_clause([-lits[i], -s[i-1][K-1]])

    # rhs[j-1] <=> sum(lits) <= j
    rhs = [s[n-1][j] for j in range(K)]

    return rhs

def amk_tot(solver, lits, K):
    if isinstance(lits, dict):
        lits = list(lits.values())
    
    top = solver.nof_vars()
    tot = ITotalizer(lits=lits, ubound=K, top_id=top)

    for c in tot.cnf.clauses:
        solver.add_clause(c)

    return tot.rhs

def add_limit_label_constraints(solver, lits, K, strategy):
    if strategy == 'nsc':
        return amk_nsc(solver, lits, K)
    elif strategy == 'sc':
        return amk_sc(solver, lits, K)
    elif strategy == 'sc_reduced':
        return amk_sc_reduced(solver, lits, K)
    elif strategy == 'nsc_reduced':
        return amk_nsc_reduced(solver, lits, K)
    elif strategy == 'tot':
        return amk_tot(solver, lits, K)

    



def solve_and_print(solver, var_map, rhs, num_labels, type):
    if type != 'incremental' and type != 'assumptions' and type != 'first':
        raise ValueError("Type must be either 'incremental', 'assumptions', or 'first'")
    if type == 'incremental':
        solver.add_clause([-rhs[num_labels - 1]])
    status = None
    if type == 'assumptions':
        status = solver.solve(assumptions = [-rhs[num_labels - 1]]) 
    else :
        status = solver.solve()
    if status:
        model = solver.get_model()
        assignment = {}
        for (i, v), varnum in var_map.items():
            if model[varnum-1] > 0:
                assignment[i] = v
        print("Solution:")
        print(assignment)
        return assignment
    else:
        print("Cannot find solution.")
        return None

def verify_solution(assignment, var, var_file, ctr_file):
    if assignment is None:
        return False
    with open(var_file) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) > 3:
                if assignment[int(parts[0])] != int(parts[2]):
                    return False

    with open(ctr_file) as f:
        for line in f:
            if line.strip() == '\x00':
                continue
            parts = line.strip().split()
            if not parts:
                continue
            i, j = int(parts[0]), int(parts[1])
            if i not in assignment or j not in assignment:
                return False
            vi = assignment[i]
            vj = assignment[j]
            if(vi not in var[i]) or (vj not in var[j]):
                return False
            if '>' in parts:
                distance = int(parts[4])
                if abs(vi - vj) <= distance:
                    print(f"\n{i} ({vi}) {j} ({vj}) <= {distance}")
                    return False
            elif '=' in parts:
                value = int(parts[4])
                if abs(vi - vj) != value:
                    return False
    return True

def main():
    start_time = time()
    helpers = "Use: python3 main.py <dataset> <num_frequencies>\n"

    dataset_folder = os.path.join("datasets", sys.argv[1])
    num_frequencies = int(sys.argv[2])
    if not os.path.exists(dataset_folder):
        print("Dataset folder does not exist.\n" + helpers)
        return
   

if __name__ == "__main__":
    main()