import sys
from counting_representations import *

# Get n from system arguments
if len(sys.argv) < 2 or sys.argv[1] not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
    print('Usage: python3 arithmetic_method_Tn.py <n = 1, ..., 10>')
    exit(0)
n = int(sys.argv[1])

# Compute the representation zeta function of the upper triangular groups G = 𝕋_n for n = 1, ... , 10
ZETA_Tn = {}

# Describe the group 𝕋_n as matrix
A = sp.Matrix([[ sp.Symbol(f'a_{{{i},{j}}}') if i <= j else 0 for j in range(n) ] for i in range(n) ])
G = TriangularGroup(System([], [], []), A)

time_before = time.perf_counter()
zeta = simplify_zeta(G.zeta_function())
time_after = time.perf_counter()
time_elapsed = time_after - time_before

ZETA_Tn[n] = zeta

print(f'({time_elapsed:.2f} sec) ζ_{{𝕋_{n}}}(s) = {str(zeta)}')
