import sympy as sp
from grothendieck_solver import System, Solver
import re
import os
import sys
import itertools
import multiprocessing

main = (__name__ == '__main__')

# Get n from system arguments
if len(sys.argv) < 2 or sys.argv[1] not in ['2', '3', '4', '5']:
    print('Usage: python3 tqft_method_Un.py <n = 2, 3, 4, 5>')
    exit(0)
n = int(sys.argv[1])

q = sp.Symbol('q')

# Class of the group
Un = q**(n * (n - 1) // 2)
Tn = q**(n * (n - 1) // 2) * (q - 1)**(n - 1)

solver = Solver()

# Util function
def compute_system(gens, cl_eqs, op_eqs):
    return solver.compute_grothendieck_class(System(gens, cl_eqs, op_eqs))

def assertion(description, value):
    if value:
        print(f'[✅] {description}')
    else:
        print(f'[❌] {description}')
        assert False

progress_text = ''

def progress_update(f, text):
    global progress_text
    progress_text = text
    percentage = int(100000 * f) / 1000
    print(f'[⏳ {percentage}%] {text}', end='\r')
    
def progress_done(text = None):
    global progress_text
    if text != None:
        progress_text = text
    print(f'[👍] {progress_text}         ')

if main:
    manager = multiprocessing.Manager()
    tasks_done = manager.Value('i', 0) # a shared variable indicates how many tasks are done
    tasks_done_lock = manager.Lock() # a lock for updating the shared variables

def compute_parallel_helper(f, args, stats):    
    # Do the computation
    value = f(*args)
    
    # Update stats
    tasks_done, tasks_done_lock, n, description = stats
    with tasks_done_lock:
        tasks_done.value += 1
        progress_update(tasks_done.value / n, description)
        
    # Return the value
    return value

def compute_parallel(f, args_array, description = ''):
    assert(main) # must be called from main process
    n = len(args_array) # number of tasks
    global tasks_done, tasks_done_lock
    with tasks_done_lock: # reset counter
        tasks_done.value = 0 
    
    with multiprocessing.Pool() as pool:
        results = [ pool.apply_async(compute_parallel_helper, (f, args, (tasks_done, tasks_done_lock, n, description))) for args in args_array ]
        values = [ r.get() for r in results ]
        progress_done(description)
    return values

def compute_non_parallel(f, args_array, description = ''):
    n = len(args_array) # number of tasks    
    values = []
    for i, args in enumerate(args_array):
        progress_update(i / n, description)
        values.append(f(*args))
    progress_done()
    return values

# Chapter 0: Define variables

# Define lots of variables
A = sp.Matrix([ [ (sp.Symbol('a_{{{},{}}}'.format(i, j)) if i < j else 1) if j >= i else 0 for j in range(n) ] for i in range(n) ])
B = sp.Matrix([ [ (sp.Symbol('b_{{{},{}}}'.format(i, j)) if i < j else 1) if j >= i else 0 for j in range(n) ] for i in range(n) ])
C = sp.Matrix([ [ (sp.Symbol('c_{{{},{}}}'.format(i, j)) if i < j else 1) if j >= i else 0 for j in range(n) ] for i in range(n) ])
D = sp.Matrix([ [ (sp.Symbol('d_{{{},{}}}'.format(i, j)) if i < n - 1 else 1) if i == j else 0 for j in range(n) ] for i in range(n) ])
X = sp.Matrix([ [ (sp.Symbol('x_{{{},{}}}'.format(i, j)) if i < n - 1 else 1) if j >= i else 0 for j in range(n) ] for i in range(n) ])
Y = sp.Matrix([ [ (sp.Symbol('y_{{{},{}}}'.format(i, j)) if i < n - 1 else 1) if j >= i else 0 for j in range(n) ] for i in range(n) ])

A_vars = [ x for x in A if x != 0 and x != 1 ]
B_vars = [ x for x in B if x != 0 and x != 1 ]
C_vars = [ x for x in C if x != 0 and x != 1 ]
D_vars = [ x for x in D if x != 0 and x != 1 ]
X_vars = [ x for x in X if x != 0 and x != 1 ]
Y_vars = [ x for x in Y if x != 0 and x != 1 ]

A_det, B_det, C_det, D_det, X_det, Y_det = A.det(), B.det(), C.det(), D.det(), X.det(), Y.det()

A_adj = sp.simplify(A.inv() * A_det)
B_adj = sp.simplify(B.inv() * B_det)
C_adj = sp.simplify(C.inv() * C_det)
D_adj = sp.simplify(D.inv() * D_det)
X_adj = sp.simplify(X.inv() * X_det)
Y_adj = sp.simplify(Y.inv() * Y_det)


# Chapter 1: Conjugacy classes

# Load representatives of unipotent conjugacy classes
unipotent_representatives = []
with open(f'data/U{n}_unipotent_representatives.txt', 'r') as file:
    for line in file:
        unipotent_representatives.append(sp.Matrix(eval(line, {'__builtins__': None}, {})))

# Let M be the number of unipotent conjugacy classes        
M = len(unipotent_representatives)

if main:
    print(f'[👉] Unipotent group of rank {n} has {M} conjugacy classes.')
    
# All representatives are unipotent_representatives
representatives = unipotent_representatives
N = M

# Find the equations defining the closure of the conjugacy class of a given representative g
def find_closed_conjugacy_equations(g):
    # Some entries of g can be variables, take those into account
    g_vars = list(set(x for x in g if x.is_Symbol))
    # Compute a general conjugate of g
    M = sp.expand(X * g * X_adj)    
    # Find equations y_i - f_i = 0
    X_det_inv = sp.Symbol('{\\det(X)^{-1}}')
    eqs = [ X.det() * X_det_inv - 1 ] # Note: it is important that the determinant of X is invertible. However, we cannot simply impose det(X) = 1, which would give too many syzygies!
    for i in range(n):
        for j in range(i, n):
            eqs.append(Y[i, j] * X.det() - M[i, j])
    
    # Since we are only looking for syzygies, we can remove all equations that uniquely contain some variable x_{i, j}
    # (and keep doing that as long as we can)
    while True:
        for x in X_vars:
            eqs_with_x = [ eq for eq in eqs if x in eq.free_symbols ]
            if len(eqs_with_x) == 1:
                eqs.remove(eqs_with_x[0])
                break
        else:
            break
                
    # Eliminate variables in X and g
    v = [ X_det_inv ] + g_vars + X_vars + Y_vars # Note: important that (g_vars and) X_vars comes before Y_vars, due to 'lex' ordering!
    grbasis = sp.groebner(eqs, *v, order = 'lex', method = 'f5b')
    
    # Relations are all equations in the Gröbner basis without any x's
    relations = [ eq for eq in grbasis if all(x not in eq.free_symbols for x in X_vars + [ X_det_inv ]) ]
        
    return relations

# Applies equations `eqs` to the matrix `M`. Note that the equations are equations in the variables Y[i, j]
def apply_equations(eqs, M):
    # Make a dictionary of substitutions
    S = { Y[i, j]: M[i, j] for i in range(n) for j in range(i, n) }
    
    # Convert equations
    return [ eq.subs(S) for eq in eqs ]

closed_conjugacy_equations = {}
closed_conjugacy_classes = {}

def compute_closed_unipotent_conjugacy_equations():
    for i in range(M):
        progress_update(i / M, 'Computing closed unipotent conjugacy equations ...')
        eqs = find_closed_conjugacy_equations(representatives[i])
        closed_conjugacy_equations[i] = eqs
    progress_done()

def compute_closed_unipotent_conjugacy_classes():
    for i in range(M):
        progress_update(i / M, 'Computing closed conjugacy classes ...')
        g = representatives[i]
        g_vars = list(set(x for x in g if x.is_Symbol))
        v = A_vars + g_vars
        eqs = closed_conjugacy_equations[i]
        cl = compute_system(v, apply_equations(eqs, A), g_vars)
        closed_conjugacy_classes[i] = cl
    progress_done()

# If results have already been computed, just read them
if os.path.isfile(f'data/U{n}_closed_conjugacy_equations.txt'):
    print('[🗂️] Reading closed conjugacy equations ...')
    
    # Read equations
    with open(f'data/U{n}_closed_conjugacy_equations.txt', 'r') as file:
        def convert_symbols(match):
            return f'{match.group(1).upper()}[{match.group(2)},{match.group(3)}]'
        s = re.sub(r'(\w)_{(\d+),(\d+)}', convert_symbols, file.read())
        exec(s)
else:
    # If they have not been computed yet, compute them and store them
    compute_closed_unipotent_conjugacy_equations()
    
    # Store equations
    with open(f'data/U{n}_closed_conjugacy_equations.txt', 'w') as file:
        file.write('closed_conjugacy_equations = {\n')
        for i in closed_conjugacy_equations:
            file.write(f' {i}: {str(closed_conjugacy_equations[i])},\n')
        file.write('}\n')
        
if os.path.isfile(f'data/U{n}_closed_conjugacy_classes.txt'):
    print('[🗂️] Reading closed conjugacy classes ...')
    
    # Read classes
    with open(f'data/U{n}_closed_conjugacy_classes.txt', 'r') as file:
        exec(file.read())
else:
    # Compute classes from equations
    compute_closed_unipotent_conjugacy_classes()
    
    # Store classes
    with open(f'data/U{n}_closed_conjugacy_classes.txt', 'w') as file:
        file.write('closed_conjugacy_classes = {\n')
        for i in closed_conjugacy_classes:
            file.write(f' {i}: {str(closed_conjugacy_classes[i])},\n')
        file.write('}\n')

# Find the ordering of the conjugacy classes:
# We say that conjugacy class i <= conjugacy class j iff the closure of i is contained in the closure of j
# Or equivalently, if the representative of class i satisfies the closed equations of class j

conjugacy_ordering = {}

def compute_unipotent_conjugacy_ordering():
    for i in range(M):
        progress_update(i / M, 'Computing ordering of conjugacy classes ...')
        js = []
        for j in range(M):
            g = representatives[j]
            # g = g.subs({ D[l, l]: l + 17 for l in range(n - 1) }) # replace diagonal entries of representative with some `random` values
            eqs = apply_equations(closed_conjugacy_equations[i], g)
            if all(eq == 0 for eq in eqs):
                js.append(j)
        conjugacy_ordering[i] = js
    progress_done()

if main:
    compute_unipotent_conjugacy_ordering()

def has_loops(graph):
    import copy
    graph = copy.deepcopy(graph) # Make deep copy since we change the graph!
    while True:
        # If the graph is empty, it has no loops!
        if len(graph) == 0:
            return False
        for x in graph:
            # If the only y -> x is x -> x, then we might as well remove x from the graph
            # Also, if the only x -> y is x -> x, we might as well remove x from the graph
            ys_to_x = [ y for y in graph if x in graph[y] ]
            if graph[x] == [x] or ys_to_x == [x]:
                for y in ys_to_x:
                    graph[y].remove(x)
                del graph[x]
                break # break to while-loop to start over
        else:
            print(graph)
            return True

if main:
    # Make sure that the ordering has no loops!
    assertion('Ordering of conjugacy classes is loop-free', not has_loops(conjugacy_ordering))

def create_transition_matrix(graph):
    N = len(graph)
    transition_matrix = sp.zeros(N, N)
    created_row = [False] * N
    
    def create_row(j):
        # If already created row, nothing to do!
        if created_row[j]:
            return
        
        # First create all columns k for which k <= j
        # Then, row[j] = e_j - \sum_{k \to j, k \ne j} row[k]
        transition_matrix[j, j] = 1
        for k in graph[j]:
            if k == j:
                continue
                
            assert transition_matrix[k, j] == 0
                
            create_row(k)
            transition_matrix[j, :] -= transition_matrix[k, :]
        
        # Mark row as created
        created_row[j] = True
    
    for j in range(N):
        create_row(j)
    
    return transition_matrix

if main:
    # Compute that transition matrix
    conjugacy_matrix = create_transition_matrix(conjugacy_ordering)

if main:
    # Compute conjugacy classes using transition matrix
    conjugacy_classes = {}
    for i in range(M):
        conjugacy_classes[i] = sp.factor(sum(conjugacy_matrix[i, j] * closed_conjugacy_classes[j] for j in range(M)))

# Compute stabilizers of the representatives
# Note that these stabilizers can be shown to be constant, so we can simply pick some values for the variable diagonals

# The orbit of any representative can be computed as the class of the group divided by the class of the stabilizer
stabilizer_classes = {}
orbit_classes = {}

def compute_orbits_stabilizers():    
    for i in range(N):
        progress_update(i / N, 'Computing stabilizers ...')
        xi = representatives[i].subs({ D[l, l]: l + 17 for l in range(n - 1) })
        eqs = list(eq for eq in sp.simplify(X * xi - xi * X) if eq != 0)
        stabilizer_classes[i] = sp.factor(compute_system(X_vars, eqs, [ X[l, l] for l in range(n - 1) ]))
    progress_done()

    for i in range(N):
        progress_update(i / N, 'Computing orbits ...')
        orbit_classes[i] = Tn / stabilizer_classes[i]
    progress_done()

if main:
    compute_orbits_stabilizers()

if main:
    # Do some checks on the transition matrix:
    # (1) The sum of all conjugacy classes should be equal to the class of the group
    # assertion('Conjugacy classes add up to group', sp.expand(q**(n * (n - 1) // 2) * (q - 1)**(n - 1) - sum(conjugacy_classes[i] for i in range(N))) == 0)

    # (1) Sum of unipotent classes should be q**(n * (n - 1) // 2)
    assertion(f'Unipotent conjugacy classes add up to q^{n * (n - 1) // 2}', sp.expand(q**(n * (n - 1) // 2) - sum([ conjugacy_classes[i] for i in range(M) ])) == 0)

    # (2) The ordering has one connected component
    assertion('Ordering has one connected component', sp.expand(sum([ conjugacy_matrix[j, i] for i in range(M) for j in range(M) ])) == 1)

    # (3) All (unipotent) conjugacy classes have positive class
    assertion('Conjugacy classes have positive leading coefficient', all(sp.LC(conjugacy_classes[i], q) > 0 for i in range(M)))

    # (4) The orbits of unipotent classes should be equal to their orbits
    assertion('Orbit unipotent conjugacy class equals conjugacy class', all(orbit_classes[i] == conjugacy_classes[i] for i in range(M)))

# Chapter 2: Compute first column of TQFT

# Need to compute the coefficients E[i, j] = [{ A \in G : [A, \xi_j] \in C_i }],
# where i and j range over the unipotent conjugacy classes

def compute_E(i, j):
    # Compute commutator [A, \xi_j]
    xi = representatives[j]
    comm = sp.simplify(A * xi * A.inv() * xi.inv())
    
    # Determine equations
    eqs = apply_equations(closed_conjugacy_equations[i], comm)
        
    # Trick for conjugacy class of identity:
    if i == 0:
        eqs = list(eq for eq in sp.simplify(A * xi - xi * A) if eq != 0)
        
    # Solve system of equations (note that the entries on the diagonal of A should be non-zero, as well as the variables on the diagonal of xi)
    cl = sp.factor(compute_system(A_vars, eqs, []))
    
    return cl

def compute_E_all():
    # Compute coefficients F[i, j, k] in parallel
    ijs = [ (i, j) for i in range(M) for j in range(N) ]
    Es = compute_parallel(compute_E, ijs, 'Computing coefficients E[i, j] ...')
    # Fill in coefficients in tensor
    for u, (i, j) in enumerate(ijs):
        E[i, j] = Es[u]

if main:
    E = sp.MutableDenseNDimArray([ 0 ] *(M * N)).reshape(M, N)

    # If results have already been computed, just read them
    if os.path.isfile(f'data/U{n}_closed_E.txt'):
        print('[🗂️] Reading coefficients E[i, j] ...')

        # Read equations
        with open(f'data/U{n}_closed_E.txt', 'r') as file:
            for line in file:
                exec(line)

    else:
        # If they have not been computed yet, compute them and store them    
        compute_E_all()

        # Store coefficients
        with open(f'data/U{n}_closed_E.txt', 'w') as file:
            for i in range(M):
                for j in range(N):
                        file.write(f'E[{i},{j}] = {str(E[i, j])}\n')

if main:
    # Now, compute the first column using the coefficients E[i, j] and the transition matrices
    first_column = sp.zeros(M, 1)
    for i in range(M):
        progress_update(i / M, 'Computing first column of TQFT ...')
        
        value = 0
        ks = [ k for k in range(M) if conjugacy_matrix[i, k] != 0 ]
        for j in range(N):
            value += sum(conjugacy_matrix[i, k] * E[k, j] * orbit_classes[j] for k in ks)
        
        first_column[i] = sp.factor(value / conjugacy_classes[i])
    progress_done()

if main:
    # Check weighted sum should add up to the class of the group squared
    assertion(f'First column (weighted) sum is group squared = {Un**2}', 0 == sp.expand(Un**2 - sum(first_column[i] * conjugacy_classes[i] for i in range(M))))

# Chapter 3: Compute other columns of TQFT

# Need to compute the coefficients F[i, j, k] = [{ g \in C_j : g \xi_k \in C_i }],
# where i, j, k range over the unipotent conjugacy classes

def compute_F(i, j, k):
    # Let g be a general unipotent element (take variables from A)
    g = sp.Matrix(A)
    for l in range(n):
        g[l, l] = 1
    g_vars = list(set([ x for x in g if x.is_Symbol ]))
    
    # Equations to solve for: g \in C_j and g \xi_k \in C_i
    eqs = apply_equations(closed_conjugacy_equations[j], g) + apply_equations(closed_conjugacy_equations[i], g * representatives[k])
    
    # Construct system from equations, and solve it
    return compute_system(g_vars, eqs, [])

def compute_F_all():
    # Compute coefficients F[i, j, k] in parallel
    ijks = [ (i, j, k) for i in range(M) for j in range(M) for k in range(M) ]
    Fs = compute_parallel(compute_F, ijks, 'Computing coefficients F[i, j, k] ...')
    # Fill in coefficients in tensor
    for u, (i, j, k) in enumerate(ijks):
        F[i, j, k] = Fs[u]

if main:
    F = sp.MutableDenseNDimArray([ 0 ] *(M ** 3)).reshape(M, M, M)

    # If results have already been computed, just read them
    if os.path.isfile(f'data/U{n}_closed_F.txt'):
        print('[🗂️] Reading coefficients F[i, j, k] ...')

        # Read equations
        with open(f'data/U{n}_closed_F.txt', 'r') as file:
            for line in file:
                exec(line)

    else:
        # If they have not been computed yet, compute them and store them    
        compute_F_all()

        # Store coefficients
        with open(f'data/U{n}_closed_F.txt', 'w') as file:
            for i in range(M):
                for j in range(M):
                    for k in range(M):
                        file.write(f'F[{i},{j},{k}] = {str(F[i, j, k])}\n')

# Now compute the matrix Z from the first column and the coefficients F[i, j, k]
# In the below function, i and j range over the unipotent conjugacy classes!

def compute_Z(i, j):
    ms = [ m for m in range(M) if conjugacy_matrix[i, m] != 0 ] # Only consider the relevant m's and l's to prevent unnecessary computations
    ls = [ l for l in range(M) if conjugacy_matrix[j, l] != 0 ]
    
    return sp.factor(sum(
        conjugacy_matrix[i, m] *
        conjugacy_matrix[j, l] *
        F[m, l, k] *
        first_column[k] *
        conjugacy_classes[k] for k in range(M) for l in ls for m in ms) / conjugacy_classes[i])

if main:
    Z = sp.zeros(M, M)
    for i in range(M):
        for j in range(M):
            progress_update((i * M + j) / (M * M), 'Computing coefficients Z[i, j] ...')
            Z[i, j] = compute_Z(i, j)
    progress_done()

if main:
    # Make sure that the first column of Z is still the first_column
    assertion('First column is unchanged', Z[:, 0] == first_column)

def matlab_to_matrix(s):
    return eval('sp.Matrix([' + s.replace('^','**').replace(';', '],[') + '])')

def matrix_to_matlab(M):
    h, w = M.shape
    return '[' + ';'.join([ ','.join([ str(M[i, j]).replace('**', '^') for j in range(w) ]) for i in range(h) ]) + ']'

if main:
    # Save Z to file
    filename = f'data/U{n}_Z.txt'
    with open(filename, 'w') as file:
        file.write(matrix_to_matlab(Z))
        
    print(f'[🏁] Saved Z to \'{filename}\'')
