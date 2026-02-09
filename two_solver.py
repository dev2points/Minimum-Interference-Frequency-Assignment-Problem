import os
# import psutil
import sys
from time import time
from pysat.solvers import Solver
from pysat.card import ITotalizer
from pysat.card import CardEnc, EncType



def create_var_map(vertices, num_frequencies):
    var_map = [[0] * (num_frequencies) for _ in range(vertices)]

    top = 0
    for i in range(vertices):
        for j in range(num_frequencies):
            top += 1
            var_map[i][j] = top 
    
    return var_map, top # array 2 sided: var_map[vertex][frequency] = varnum
    

def add_constraints(solver, num_frequencies, ctr_file, type_eo):
    penalty_var_map = {}
    penalty_values = {}
    with open(ctr_file) as f:
        header = f.readline().strip().split()
        vertices = int(header[2])
        edges = int(header[3])
        var_map, top = create_var_map(vertices, num_frequencies)
        top = add_eo(solver, var_map, vertices, num_frequencies, top, type_eo)
        penalty_var_map = {}
        for line in f:
            if line.strip() == '\x00':
                continue
            parts = line.strip().split()
            if not parts:
                continue
            u, v = int(parts[1]), int(parts[2])
            distance = int(parts[3])
            penalty = int(parts[4])
            top +=1
            penalty_var_map[(u,v)] = top
            penalty_values[(u,v)] = penalty
            # print(f"distance for edge ({u},{v}) = {distance}, penalty = {penalty}")
            for i in range(num_frequencies):
                j_min = max(0, i - distance)
                j_max = min(num_frequencies - 1, i + distance)
                for j in range(j_min, j_max + 1):
                    solver.add_clause([
                        -var_map[u][i],
                        -var_map[v][j],
                        top
                    ])
                    # print(f"add clause: varmap[{u}][{i}]={var_map[u][i]} ^ varmap[{v}][{j}]={var_map[v][j]} -> penalty_var={top}")

            # violation_vars = []

            # # tạo biến phụ c_ij
            # for i in range(num_frequencies):
            #     j_min = max(0, i - distance)
            #     j_max = min(num_frequencies - 1, i + distance)

            #     for j in range(j_min, j_max + 1):
            #         top += 1
            #         c_ij = top
            #         violation_vars.append(c_ij)

            #         # c_ij ↔ (x_u_i ∧ x_v_j)
            #         solver.add_clause([-c_ij, var_map[u][i]])
            #         solver.add_clause([-c_ij, var_map[v][j]])
            #         solver.add_clause([
            #             -var_map[u][i],
            #             -var_map[v][j],
            #             c_ij
            #         ])

            #         # c_ij ⇒ p_uv
            #         solver.add_clause([-c_ij, penalty_var_map[(u, v)]])

            # # p_uv ⇒ OR(c_ij)
            # # (¬p_uv ∨ c1 ∨ c2 ∨ ... ∨ ck)
            # solver.add_clause([-penalty_var_map[(u, v)]] + violation_vars)

    if edges != len(penalty_var_map):
        raise ValueError(
            f"Mismatch: edges={edges}, penalty_vars={len(penalty_var_map)}"
    )
    # print(f"edges:{edges}, length penalty vars:{len(penalty_var_map)}")


    # vertices: number of vertices
    # var_map: array 2 sided perform SAT variable: var_map[vertex][frequency]
    # penalty_var_map: dict perform SAT variable: penalty_var_map((u,v), var)
    # penalty_value: dict perform value of penalty of edge: penalty_value((u,v), value)
    return vertices, var_map, penalty_var_map, penalty_values
            

def eo_pairwise(solver, var_map, vertices, num_frequencies, top):
    for i in range(vertices):
        solver.add_clause([var_map[i][j] for j in range(num_frequencies)]) # at least one
        for j in range(num_frequencies - 1):
            for k in range(j + 1, num_frequencies):
                solver.add_clause([-var_map[i][j], -var_map[i][k]])
    return top


def eo_sc(solver, var_map, vertices, num_frequencies, top):
    for i in range(vertices):

        # AT LEAST ONE
        solver.add_clause([var_map[i][j] for j in range(num_frequencies)])

        # auxiliary vars s[0..F-2]
        s = {}
        for j in range(num_frequencies):
            top += 1
            s[j] = top

        # (¬x0 ∨ s0)
        solver.add_clause([-var_map[i][0], s[0]])

        for j in range(1, num_frequencies - 1):
            # (¬xj ∨ sj)
            solver.add_clause([-var_map[i][j], s[j]])

            # (¬s_{j-1} ∨ sj)
            solver.add_clause([-s[j - 1], s[j]])

            # (¬xj ∨ ¬s_{j-1})
            solver.add_clause([-var_map[i][j], -s[j - 1]])

        # (¬x_{F-1} ∨ ¬s_{F-2})
        solver.add_clause([
            -var_map[i][num_frequencies - 1],
            -s[num_frequencies - 2]
        ])

    return top

def eo_card(solver, var_map, vertices, num_frequencies, top,
            encoding):
    """
    encoding:
        EncType.seqcounter   (khuyến nghị)
        EncType.pairwise
        EncType.totalizer
        EncType.cardnetwrk
        ...
    """
    encoding = int(encoding.split('_')[-1])
    if encoding < 0 or encoding > 9:
        raise ValueError(
            "Card type:\n"
            "    pairwise    = 0\n"
            "    seqcounter  = 1\n"
            "    sortnetwrk  = 2\n"
            "    cardnetwrk  = 3\n"
            "    bitwise     = 4\n"
            "    ladder      = 5\n"
            "    totalizer   = 6\n"
            "    mtotalizer  = 7\n"
            "    kmtotalizer = 8\n"
            "    native      = 9"
        )
    for i in range(vertices):
        lits = [var_map[i][j] for j in range(num_frequencies)]

        card = CardEnc.equals(
            lits=lits,
            bound=1,
            encoding=encoding,
            top_id=top
        )

        top = card.nv  # cập nhật biến phụ lớn nhất

        for c in card.clauses:
            solver.add_clause(c)

    return top

from pysat.pb import PBEnc

def eo_pb(solver, var_map, vertices, num_frequencies, top,
          encoding):
    """
    encoding:
        EncType.seqcounter   (khuyến nghị)
        EncType.pairwise
        EncType.totalizer
        EncType.cardnetwrk
        ...
    """
    encoding = int(encoding.split('_')[-1])
    if encoding < 0 or encoding > 9:
        raise ValueError(
            "PB type:\n"
            "    pairwise    = 0\n"
            "    seqcounter  = 1\n"
            "    sortnetwrk  = 2\n"
            "    cardnetwrk  = 3\n"
            "    bitwise     = 4\n"
            "    ladder      = 5\n"
            "    totalizer   = 6\n"
            "    mtotalizer  = 7\n"
            "    kmtotalizer = 8\n"
            "    native      = 9"
        )

    for i in range(vertices):
        lits = [var_map[i][j] for j in range(num_frequencies)]
        weights = [1] * num_frequencies

        pb = PBEnc.equals(
            lits=lits,
            weights=weights,
            bound=1,
            encoding=encoding,
            top_id=top
        )

        top = pb.nv  # cập nhật biến phụ

        for c in pb.clauses:
            solver.add_clause(c)

    return top


def eo_nsc(solver, var_map, vertices, num_frequencies, top):
    for i in range(vertices):

        r = {}
        for j in range(num_frequencies):
            top += 1
            r[j] = top

        for j in range(num_frequencies):
            solver.add_clause([-var_map[i][j], r[j]])
        for j in range(1, num_frequencies):
            solver.add_clause([-r[j-1], r[j]])
            solver.add_clause([var_map[i][j], r[j-1], -r[j]])

        solver.add_clause([var_map[i][0], -r[0]])
        solver.add_clause([r[num_frequencies-1]]) # at least one

        for j in range(1, num_frequencies):
            solver.add_clause([-var_map[i][j], -r[j-1]]) # at most one

        # #at most one
        # for j in range(1, num_frequencies):
        #     solver.add_clause([-var_map[j], -r[j-1]])

    return top


def add_eo(solver, var_map, vertices, num_frequencies, top, strategy):
    if strategy == "nsc":
        return eo_nsc(solver, var_map, vertices, num_frequencies, top)
    elif strategy == "sc":
        return eo_sc(solver, var_map, vertices, num_frequencies, top)
    elif strategy == "pairwise":
        return eo_pairwise(solver, var_map, vertices, num_frequencies, top)
    elif strategy.startswith("card"):
        return eo_card(solver, var_map, vertices, num_frequencies, top, strategy)
    elif strategy.startswith("pb"):
        return eo_pb(solver, var_map, vertices, num_frequencies, top, strategy)
    



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

def add_limit_total_penalty(solver, lits, K, strategy):
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





def solve_and_print(solver, var_map, rhs, total_penalty, type):
    if type not in ('incremental', 'assumptions', 'first'):
        raise ValueError("Type must be either 'incremental', 'assumptions', or 'first'")

    if type == 'incremental':
        solver.add_clause([-rhs[total_penalty - 1]])

    if type == 'assumptions':
        status = solver.solve(assumptions=[-rhs[total_penalty - 1]])
    else:
        status = solver.solve()

    if not status:
        print("Cannot find solution.")
        return None

    model = solver.get_model()
    assignment = {}
    for i in range(len(var_map)):
        for v in range(len(var_map[i])):
            varnum = var_map[i][v]
            if varnum > 0 and model[varnum - 1] > 0:
                if assignment.get(i) is not None:
                    raise ValueError(f"Vertex {i} has multiple assigned frequencies")
                assignment[i] = v

    # print("Solution:")
    # print(assignment)


    

    return assignment


def check_solution(ctr_file, assignment, vertices):

    total_penalty = 0

    with open(ctr_file, 'r') as f:
        header = f.readline().strip().split()
        edges = int(header[3])

        for line in f:
            if not line.strip() or line.strip() == '\x00':
                continue

            parts = line.strip().split()
            # if parts[0] != 'e':
            #     continue

            freq_count = {}

            for v, f in assignment.items():
                freq_count[v] = freq_count.get(v, 0) + 1

            for v in range(vertices):
                if v not in freq_count:
                    raise ValueError(f"Vertex {v} has ZERO frequency")
                if freq_count[v] != 1:
                    raise ValueError(f"Vertex {v} has {freq_count[v]} frequencies")
                

            u = int(parts[1])
            v = int(parts[2])
            d = int(parts[3])
            p = int(parts[4])

            fu = assignment[u]
            fv = assignment[v]

            if abs(fu - fv) <= d:
                if p == 0:
                    raise ValueError(f"Penalty = 0 with constraint {u, v, d}")
                else:
                    total_penalty += p

    print(f"Total penalty: {total_penalty}")
    return total_penalty


def main():
    start_time = time()
    helpers = "Use: python3 main.py <dataset> <num_frequencies> <type_eo> <type_amk> <solver> <strategy>\n"

    dataset = os.path.join("datasets", sys.argv[1] + ".ctr.txt")
    num_frequencies = int(sys.argv[2])
    type_eo = sys.argv[3]
    type_amk = sys.argv[4]
    if not os.path.exists(dataset):
        print("Dataset folder does not exist.\n" + helpers)
        return
    solver = Solver(name = sys.argv[5])
    vertices, var_map, penalty_var_map, penalty_values = add_constraints(solver, num_frequencies, dataset, type_eo)
    
    assignment = solve_and_print(solver, var_map,None, None, 'first')
    if assignment is None:
        return
    total_penalty = check_solution(dataset, assignment, vertices)
    print(f"Time: {time() - start_time}")
    
    solver.delete()

    solver = Solver(name="cadical195")
    vertices, var_map, penalty_var_map, penalty_values = add_constraints(solver, num_frequencies, dataset, type_eo)

    rhs = add_limit_total_penalty(solver, penalty_var_map, total_penalty, type_amk)

    while total_penalty > 0 :
        print("--------------------------------------------------")
        print(f"\nTrying with at most {total_penalty - 1} penalty...")
        assignment= solve_and_print(solver, var_map, rhs, total_penalty, sys.argv[6])
        if assignment is None:
            return
        total_penalty =  check_solution(dataset, assignment, vertices)
        print(f"Time: {time() - start_time}")
   

if __name__ == "__main__":
    main()