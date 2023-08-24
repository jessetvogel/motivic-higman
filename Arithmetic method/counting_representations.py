#!/usr/bin/env python
# coding: utf-8

# In[79]:


import time
import sympy as sp
from IPython.display import display, Math
from grothendieck_solver import System, Solver
from zeta import simplify_zeta


# #### Symbols and definitions

# In[89]:


# Define common symbols
ZERO, ONE = sp.sympify(0), sp.sympify(1)
q, s = sp.symbols('q s')

# Create a solver instance
solver = Solver()

# Define constants
MAX_DEPTH = 20
FLAG_DEBUG = False


# #### Utilities

# In[67]:


# Input:
# - a list `gens = [x1, ... , xn]` of variables
# - an expression `expr = a1 * x1 + ... + an * xn`
# Output:
# the list `[a1, ... , an]` of coefficients
def get_linear_coefficients(gens, expr):
    if expr == 0:
        return [ 0 ] * len(gens)
    
    assert sp.total_degree(expr, *gens) == 1, f'The expression `{expr}` is not linear in {gens}!'
    return sp.reduced(expr, gens)[0]


# In[68]:


# Input:
# - a System `T`
# - a expression/function `f` on `T`
# Output:
# - returns `True` if it can easily be shown that the function `f` is invertible on `T`, otherwise returns `False`
#   (Note: might return `False` even though the function is invertible)
def is_invertible(T, f):
    return all (g == 1 or g == -1 or g in T.op_eqs or -g in T.op_eqs for g in get_factors(f))

def get_factors(f):
    f = sp.factor(f)
    if f.is_Mul:
        return [ g for a in f.args for g in get_factors(a) ]
    elif f.is_Pow:
        return get_factors(f.args[0])
    else:
        return [ f ]


# In[69]:


# Input:
# - a list `gens` of variables
# - a list `eqs` of equations in the variables `gens`
# Output:
# - a copy of `M`, simplified with respect to the given equations
def simplify_matrix(gens, eqs, M):
    m, n = M.shape
    M = sp.Matrix([[ sp.expand(M[i, j]) for j in range(n) ] for i in range(m) ]) # make a copy!
    if not gens or not eqs:
        return M # no reduction need be done
    for i in range(m):
        for j in range(n):
            M[i, j] = sp.reduced(M[i, j], [ eq for eq in eqs if eq != 0 ], gens)[1] # NOTE: should `eqs` be a Gröbner basis ?
    return M


# In[70]:


# A class to keep track of substitutions of variables.
# Such substitutions can be applied of expressions, and be composed to give new substitutions.
class Substitution:
    
    def __init__(self, subs = None):
        self.subs = subs
        
    def compose(self, other):
        if other.subs == None:
            return self
        if self.subs == None:
            return other
        keys = set(self.subs).union(other.subs)
        return Substitution({
            x: other.apply(self.subs[x]) if x in self.subs else other.subs[x] for x in keys
        })
    
    def apply(self, X):
        if self.subs == None:
            return X
        return X.xreplace(self.subs)
    
    def modulo(self, f):
        if self.subs == None:
            return self
        return Substitution({
            x: sp.reduced(self.subs[x], [ f ])[1] for x in self.subs
        })


# In[71]:


# Input:
# - a list `gens` of variables
# - an expression `eq` linear in the variables `gens`
# - a variable `x` (from `gens`) to solve for in `eq`
# - an expression (or matrix of expressions) `X` to be transformed, that is,
#   we want to substitute `x` for the other variables in `gens`, assuming `eq = 0`
# Output:
# - a pair `(Y, subs)` where
#   - `Y` is the transformed object
#   - `subs` is a Substitutions object describing the substitutions applied in the process
def apply_linear_eq(gens, eq, x0, X):
    # Write `eq = x0 * f0 + x1 * f1 + ... + xk * fk = 0` with `xi` in `gens` and `fi` coefficients
    # Then `x0 = - (x1 * f1 + ... + xk * fk) / f0`, so
    # replace `xi = \tilde{xi} * f0` for all `i = 1, ... , k` to make everything integral again
    # Note: writing `fi / f0 = u / v` (in reduced form), we can also replace `xi = \tilde{xi} * v`
    f0 = sp.reduced(eq, [ x0 ])[0][0]
    
    # If f0 = \pm 1, we don't need a Substitution object
    if f0 in [1, -1]:
        return (X.replace(x0, sp.expand(x0 * f0 - eq) / f0), Substitution())
    
    subs = {}
    for term in sp.Poly(eq, gens).terms():
        xi = gens[term[0].index(1)]
        fi = term[1]
        if xi != x0:
            subs[xi] = xi * sp.fraction(fi / f0)[1]
    subs = Substitution(subs)
    new_X = subs.apply(X).subs(x0, sp.expand(x0 * f0 - eq))
    return (new_X, subs)


# In[72]:


# Input:
# - a System `T`
# - a list `gens` of variables
# - a list `eqs` of equations which are linear in `gens`
# - an expression (or matrix of expressions) `X` to be transformed, as in `apply_linear_eq`
# Output:
# - a list of triples `(T', Y, subs)` where
#   - `T'` is a subvariety of `T` over which:
#   - `Y` is transformed object
#   - `subs` is a Substitutions object describing the substitutions applied in the process

def apply_linear_eqs(T, gens, eqs, X, subs = None):
    # Initialize subs if None
    if subs == None:
        subs = Substitution()
    
    # First try to solve all equations which can be solved directly (i.e. where no case distinction is needed)
    while True:
        # (remove all trivial equations)
        eqs = [ eq for eq in eqs if eq != 0 ]
        for eq in eqs:
            (x0, f0) = (None, None)
            for x in gens:
                f = sp.reduced(eq, [ x ], gens)[0][0]
                if x in eq.free_symbols and is_invertible(T, f):
                    (x0, f0) = (x, f)
                    break
            else:
                # assert False, "Cannot solve for any variable in " + str(eq)
                continue
                
            # Convert X and keep track of substitutions made
            X, new_subs = apply_linear_eq(gens, eq, x0, X) # (indeed, the coefficient `f0` is invertible)
            
            # Also apply the substitutions to each equation
            # Note: `new_subs` does not account for x0 -> x0_value!
            x0_value = sp.expand(sp.factor(new_subs.apply(sp.expand(x0 * f0 - eq) / f0))) 
            eqs = [ new_subs.apply(e).subs(x0, x0_value) for e in eqs if e != eq ]
            
            # Update subs
            subs = subs.compose(new_subs)
            break
        else:
            # If no more solutions can be solved for (directly), break
            break
            
    # If all equations are solved for, we are done!
    if not eqs:
        return [(T, X, subs)]
    
    # Otherwise, we do a case distinction.
    # Write `eq = f * x + g`
    eq = eqs[0]
    x = [ y for y in gens if y in eq.free_symbols ][0]
    [ f ], g = sp.reduced(eq, [ x ])
    cases = []
    # (1) If f = 0, then continue
    T_1 = System(T.gens, T.eqs + [ f ], T.op_eqs)
    eqs_1 = [ sp.reduced(eq, [ f ])[1] for eq in eqs ] # simplify equations
    cases.extend(apply_linear_eqs(T_1, gens, eqs_1, X, subs))
    
    # (2) If f != 0, then solve for x = - g / f
    T_2 = System(T.gens, T.eqs, T.op_eqs + [ f ])
    X_2, more_subs = apply_linear_eq(gens, eq, x, X) # (indeed, the coefficient `f` is invertible)
    eqs_2 = [ more_subs.apply(eq) for eq in eqs[1:] ]
    subs_2 = subs.compose(more_subs)
    cases.extend(apply_linear_eqs(T_2, gens, eqs_2, X_2, subs_2))
    return cases


# In[73]:


# Input:
# - a System `T`
# - a matrix `M` of functions on `T`
# - `k`
# - `A`
# - `A_inv`
# Output:
# - A tuple `(T', k, A, A_inv)` where
#   - `T'` is a subvariety of `T` on which the following hold:
#   - `k` is the number of independent columns of `M`
#   - `A` contains `k` linearly independent columns
#   - `A_inv` is the inverse of `A`

def find_independent_columns(T, M, k = 0, A = None, A_inv = None):    
    # display(Math('A = ' + sp.latex(A) + ', M = ' + sp.latex(M)))
    M = simplify_matrix(T.gens, T.eqs, M) # Reduce `M` w.r.t. `T` (this also makes a copy of `M`)
    
    m, n = M.shape
    ijs = [ (i, j) for i in range(k, m) for j in range(k, n) ]
        
    (i0, j0, i0j0_invertible) = (None, None, None)
    if A == None:
        A = sp.eye(m)
    if A_inv == None:
        A_inv = sp.eye(m)
    
    for i, j in ijs:
        # First try to find an entry on which we don't need case distinctions!
        if is_invertible(T, M[i, j]):
            (i0, j0, i0j0_invertible) = (i, j, True)
            break
    else:
        for i, j in ijs:
            if M[i, j] != 0 and not M[i, j].is_Number:
                (i0, j0, i0j0_invertible) = (i, j, False)
                break
    
    if (i0, j0) == (None, None):
        if not all(M[i, j] == 0 for i, j in ijs):
            assert False, "No suitable column found!"
        else:
            return [ (T, k, A, A_inv) ]
    
    eq = M[i, j]
    cases = []
    if not i0j0_invertible:
        # Case `eq = 0`:
        new_T = System(T.gens, T.eqs + [ eq ], T.op_eqs)
        cases.extend(find_independent_columns(new_T, M, k, A, A_inv))
                
    # Case `eq != 0`:
    if i0 != k:
        A = A.elementary_col_op('n<->m', i0, k)
        A_inv = A_inv.elementary_row_op('n<->m', i0, k)
        M = M.elementary_row_op('n<->m', i0, k)
        # display(Math('A = ' + sp.latex(A) + ', M = ' + sp.latex(M)))
        
    if j0 != k:
        M = M.elementary_col_op('n<->m', j0, k)
        # display(Math('A = ' + sp.latex(A) + ', M = ' + sp.latex(M)))

    Mkk = M[k, k]
    A = A.elementary_col_op('n->kn', k, Mkk)
    A_inv = A_inv.elementary_row_op('n->kn', k, 1 / Mkk)
    M = M.elementary_row_op('n->kn', k, 1 / Mkk) # Make M[k, k] = 1
    
    for j in range(k + 1, n): # make columns integral again
        if sp.fraction(M[k, j])[1] != 1:
            M = M.elementary_col_op('n->kn', j, Mkk) # Note: this doesn't change A
            # display(Math('A = ' + sp.latex(A) + ', M = ' + sp.latex(M)))
    for i in range(k + 1, m): # make entries below M[k, k] zero
        if M[i, k] != 0:
            A = A.elementary_col_op('n->n+km', k, M[i, k], i)
            A_inv = A_inv.elementary_row_op('n->n+km', i, - M[i, k], k)
            M = M.elementary_row_op('n->n+km', i, - M[i, k], k)
            # display(Math('A = ' + sp.latex(A) + ', M = ' + sp.latex(M)))
    for j in range(k + 1, n): # make entries right of  M[k, k] zero
        if M[k, j] != 0:
            M = M.elementary_col_op('n->n+km', j, - M[k, j], k)
            # display(Math('A = ' + sp.latex(A) + ', M = ' + sp.latex(M)))

    new_T = T if eq in [1, -1] else System(T.gens, T.eqs, T.op_eqs + [ eq ])
    cases.extend(find_independent_columns(new_T, M, k + 1, A, A_inv))
    
    return cases


# In[74]:


# Input:
# - a matrix `H`
# Output:
# - an upper triangular matrix `P * H * P^(-1)` where `P` is a permutation matrix
#   (if this is not possible, fail)
def permute_matrix_to_upper_triangular(H):
    (n, n) = H.shape
    ordered = []
    to_order = list(range(n))
    while to_order:
        for j in to_order:
            if all(i == j or H[i, j] == 0 for i in to_order):
                ordered.append(j)
                to_order.remove(j)
                break
        else:
            assert False, "Cannot be brought to upper triangular form using permutations"
    P = sp.Matrix([[ 1 if j == ordered[i] else 0 for j in range(n) ] for i in range(n) ])
    return P * H * P.inv()


# In[90]:


# A class to, very naively, store intermediate results, to be used again later
class LookupTable:
    
    def __init__(self):
        self.table = {}
        
    def put(self, G, value):
        self.table[tuple(G)] = value
        
    def get(self, G):
        t = tuple(G)
        if t not in self.table:
            return None
        
        return self.table[t]
    
LOOKUP_TABLE = LookupTable()


# #### Algorithm to find representatives and their stabilizers

# In[86]:


# Input:
# - a System `T`
# - a list `H_symbols` of variables
# - a matrix `H` of functions on `T` in terms of `H_symbols`. This encodes an (upper triangular) group H over `T`
# - a System `X` describing the variety of characters on which `H` acts
# - a row vector `chi` representing a character
# Output:
# - a list of tuples `(T', chi, stab, index)`, where
#   - `T'`: a variety over `T`
#   - `chi`: a family of representatives for an orbit // Note: we don't really need `chi` as output, but useful for debugging
#   - `stab`: a Substition object of solutions of `H_symbols` in order to obtain the stabilizer
#   - `index`: a polynomial in q which is the index of the stabilizer in H

# ? (Note that the index of the stabilizer in H is the length of the dict `stabilizer`) ?

def find_orbit_stabilizer_representatives(T, H_symbols, H, X, chi, stab = None, depth = 0):
    # display(Math('[ H = ' + sp.latex(H) + ' \\text{ over } ' + T.latex() + ' \\text{ acting on } \\chi = ' + sp.latex(chi) + ' ] \\textup{ with } X = ' + X.latex()))  
    
    # Create a stabilizer substitution if there is none yet
    if stab == None:
        stab = Substitution()
            
    if depth > MAX_DEPTH:
        # print('Maximum-depth reached for:')
        # print('T = ' + str(T))
        # print('H_symbols = ' + str(H_symbols))
        # print('H = ' + str(H))
        # print('chi = ' + str(chi))
        # print('stab = ' + str(stab))
        # display(Math('H = ' + sp.latex(H) + ' \\text{ in } ' + sp.latex(H_symbols) + ' \\text{ over } ' + T.latex() + ' \\text{ acting on } \\chi = ' + sp.latex(chi)))  
        assert False, "Maximum depth reached!"

    # Compute the image of chi (H acts by right-multiplication)
    im_chi = simplify_matrix(T.gens + X.gens, T.eqs + X.eqs, chi * H)
    
    # display(Math('[ \\chi = ' + sp.latex(chi) + ' \\overset{H}{\\mapsto} ' + sp.latex(im_chi) + ' ]'))
        
    # Starting at the back, we try to make as many entries of `chi` equal to 0 or 1
    for α, im_α in zip(reversed(chi), reversed(im_chi)):
        # If `α` is invariant under `H`, continue
        if sp.expand(im_α - α) == 0:
            continue
        
        # Note: we cannot have `α == 0` or `α == 1` anymore, those should have been invariant!
        assert α != 0 and α != 1, "Mistake in stabilizer: " + str(α) + " ↦ " + str(im_α)
                
        # Write `α ↦ u * α + v`
        try:
            u, v = sp.Poly(im_α, α).all_coeffs()
        except Exception as e:
            print(f'α = {α}, im_α = {im_α}')
            raise e
                
        # If `v = 0`, we must have `u ≠ 1` (as `α` is not invariant under `H`), and we stratify `X` based on `α`:
        #   - Case `α = 0`. Then `α` is also invariant under `H`.
        #   - Case `α ≠ 0`. Then we can use the action of H to get `α = 1`. The stabilizer should have `u = 1`.
        if v == 0:
            assert u != 1, "`" + str(α) + " ↦ " + str(im_α) + "` is both invariant and not invariant!"
            assert u in H_symbols, f"`{u}` is expected to be a diagonal entry of `H`"

            cases = []

            # Case `α = 0`
            update_stab = Substitution({ α: ZERO })
            new_stab = stab.compose(update_stab)
            
            new_X_eqs = [ eq for eq in [ sp.expand(eq.subs(α, ZERO)) for eq in X.eqs ] if eq != 0]
            new_X_op_eqs = list(set(sp.expand(eq.subs(α, ZERO)) for eq in X.op_eqs))
            new_X = System([ β for β in X.gens if β != α ], new_X_eqs, new_X_op_eqs)

            new_chi = chi.subs(α, ZERO)
            new_chi = simplify_matrix(T.gens + new_X.gens, T.eqs + new_X.eqs, new_chi)

            new_H = update_stab.apply(H)
            new_H = simplify_matrix(T.gens + new_X.gens, T.eqs + new_X.eqs, new_H)
            
            cases.extend(find_orbit_stabilizer_representatives(T, H_symbols, new_H, new_X, new_chi, new_stab, depth + 1))
            
            # Case `α ≠ 0`
            update_stab = Substitution({ α: ONE, u: ONE })
            new_stab = stab.compose(update_stab)
            
            new_X_eqs = [ eq for eq in [ sp.expand(eq.subs(α, ONE)) for eq in X.eqs ] if eq != 0]
            new_X_op_eqs = list(set(sp.expand(eq.subs(α, ONE)) for eq in X.op_eqs))
            new_X = System([ β for β in X.gens if β != α ], new_X_eqs, new_X_op_eqs)

            new_chi = chi.subs(α, ONE)
            new_chi = simplify_matrix(T.gens + new_X.gens, T.eqs + new_X.eqs, new_chi)
            
            new_H = update_stab.apply(H)
            new_H = simplify_matrix(T.gens + new_X.gens, T.eqs + new_X.eqs, new_H)
            
            new_H_symbols = [ x for x in H_symbols if x != u ] # remove u from the H_symbols
            
            # Keep track of the index `[H : new_H] = q - 1`
            for (sy, ch, st, idx) in find_orbit_stabilizer_representatives(T, new_H_symbols, new_H, new_X, new_chi, new_stab, depth + 1):
                cases.append((sy, ch, st, (q - 1) * idx))
            
            return cases
        
        # Write v = a0 * f0 + a1 * f1 + ... + ak * fk, with ai ∈ H_symbols and fi ∈ k[X]
        v_poly = sp.Poly(v, H_symbols)
            
        # Find pair (ai, fi) such that fi is invariant under the action of H
        # [⚠️ WARNING] Not sure if this is always possible!
        (ai, fi) = (None, None)
        repl = { β: im_β for β, im_β in zip(chi, im_chi) if β.is_Symbol }
        for term in v_poly.terms():
            f = term[1]
            im_f = f.xreplace(repl)
            df = sp.expand(im_f - f)
            if T.gens:
                df = sp.reduced(df, T.eqs, T.gens)[1]
            if df == 0:
                a = H_symbols[term[0].index(1)]
                # LITTLE HACK: also require that `a` does not appear in `im_β` for any `β` (not equal to `α`) which also appears in `im_α`
                if any(a in im_β.free_symbols for β, im_β in zip(chi, im_chi) if β.is_Symbol and β != α and β in im_α.free_symbols):
                    continue
                (ai, fi) = (a, f)
                break
        else:
            continue
        
        # Now, stratify X based on `fi`.
        cases = []
        
        #  (1) Case `fi ≠ 0`. Choose representatives with `α = 0`.
        #      In this case, (α = 0) ↦ u * (α = 0) + (v = a0 * f0 + a1 * f1 + ... + ak * fk),
        #      so the new stabilizer `H` wis given by `v = a0 * f0 + a1 * f1 + ... + ak * fk = 0`.
        #      We impose this condition by solving for `ai = - (a1 * f1 + ... (not ai * fi) ... + ak * fk) / fi`.
        #      However, since we don't want to divide by fi (we don't like rational functions),
        #      we actually reparametrize `aj = \tilde{aj} * fi` for all j such that `fj ≠ 0`, so that
        #      `ai = - (\tilde{a1} * f1 + ... (not ai * fi) ... + \tilde{ak} * fk)`.
        #      Furthermore, since we fixed `α = 0`, we can/should omit `α` from X.gens
        update_stab = Substitution({ α: ZERO, ai: sp.expand(ai * fi - v), **{
            aj: aj * fi for aj in v.free_symbols.intersection(H_symbols) if aj != ai
        }})
                
        new_X_eqs = [ eq for eq in [ sp.expand(eq.subs(α, ZERO)) for eq in X.eqs ] if eq != 0]
        new_X_op_eqs = list(set([ eq.subs(α, ZERO) for eq in X.op_eqs ] + [ fi ]))
        new_X = System([ β for β in X.gens if β != α ], new_X_eqs, new_X_op_eqs)
        
        new_chi = chi.subs(α, ZERO)
        new_chi = simplify_matrix(T.gens + new_X.gens, T.eqs + new_X.eqs, new_chi)
        new_stab = stab.compose(update_stab)
        
        new_H = update_stab.apply(H)
        new_H = simplify_matrix(T.gens + new_X.gens, T.eqs + new_X.eqs, new_H)
        new_H_symbols = [ a for a in H_symbols if a != ai ] # remove the a_i from H_symbols
        
        # Keep track of the index `[H : new_H] = q`
        for (sy, ch, st, idx) in find_orbit_stabilizer_representatives(T, new_H_symbols, new_H, new_X, new_chi, new_stab, depth + 1):
            cases.append((sy, ch, st, q * idx))
        
        #  (2) Or `fi = 0`. In this case we just add it as an equation and repeat.
        #      (Note: if fi is invertible, we don't need to consider this case)
        if not is_invertible(X, fi):
            new_X = System(X.gens, X.eqs + [ fi ], X.op_eqs)
            
            new_chi = simplify_matrix(T.gens + new_X.gens, T.eqs + new_X.eqs, chi)
            
            new_H = simplify_matrix(T.gens + new_X.gens, T.eqs + new_X.eqs, H)
            
            new_stab = stab.modulo(fi)
        
            # print(f'Case 2 ({f0} = 0)')
            cases.extend(find_orbit_stabilizer_representatives(T, H_symbols, new_H, new_X, new_chi, new_stab, depth + 1))
        
        return cases
    
    # At this point, chi should be invariant under H
    assert chi == im_chi, 'Unexpectedly, chi ≠ im_chi'
    
    # If chi is completely invariant, then H is the stabilizer (i.e. further no equations)
    new_T = System(T.gens + X.gens, T.eqs + X.eqs, T.op_eqs + X.op_eqs)
    return [(new_T, chi, stab, 1)]


# #### Algorithm to compute the representation zeta function of a triangular group

# In[87]:


# A class representing a family G of connected algebraic groups of upper triangular matrices,
# parametrized by a variety `T`

class TriangularGroup:
    
    def __init__(self, T, G, depth = 0):
        self.T = T
        self.G = G
        self.G_symbols = list(G.free_symbols.difference(T.gens))
        self.multiplier = 1
        self.depth = depth
        self.ident = '\\quad' * depth # for displaying math in debug mode
                        
    def display_math(self, math):
        display(Math(self.ident + ' ' + math))
    
    def simplify_presentation(self):
        self.T.simplify_equations()

        # If the GCD of all coefficients in front of some G_symbol x is invertible,
        # then make a substitution x = \tilde{x} / GCD
        gcd_changes = False
        for x in self.G_symbols:
            # Coefficients in front of x can be obtained by differentiating G w.r.t. x
            gcd = sp.factor(sp.gcd(list(self.G.diff(x))))
            if gcd == 0 or gcd == 1:
                continue
            gcd_factors = gcd.args if gcd.is_Mul else [ gcd ]
            for f in gcd_factors:
                if is_invertible(self.T, f):
                    self.G = self.G.subs(x, x / f)
                    gcd_changes = True
        
        # This seems inefficient, but needs to be done in order to clear up any mess made by the above
        if gcd_changes:
            (n, n) = self.G.shape
            for i in range(n):
                for j in range(i + 1, n):
                    self.G[i, j] = sp.expand(sp.factor(self.G[i, j]))
        
        # Factor `T` if possible. All variables/equations of which G is independent can be solved beforehand
        gens, eqs, op_eqs = [], [], []
        used_symbols = self.G.free_symbols.intersection(self.T.gens)
        for sub_T in self.T.factor():
            if used_symbols.isdisjoint(sub_T.gens):
                self.multiplier *= solver.compute_grothendieck_class(sub_T)
            else:
                gens.extend(sub_T.gens)
                eqs.extend(sub_T.eqs)
                op_eqs.extend(sub_T.op_eqs)
        self.T = System(gens, eqs, op_eqs)
        self.T.simplify_equations()

    def zeta_function(self):
        try:
            # First, simplify the presentation
            self.simplify_presentation()

            # Short-cut
            if self.multiplier == 0:
                return 0

            (n, n) = self.G.shape
            
            # If n = 0, then G is trivial
            if n == 0:
                return self.multiplier
            
            # If G_{n, n} != 1, then scale the whole matrix by 1 / G_{n, n}
            # After a suitable change of variables, we again have a simplified matrix but now with G_{n, n} = 1
            if self.G[n - 1, n - 1] != 1:
                d = self.G[n - 1, n - 1]
                assert d.is_Symbol
                self.G = (self.G / d).xreplace({ x: x / d if x != d else 1 / d for x in self.G_symbols })
                for i in range(n):
                    for j in range(i, n):
                        self.G[i, j] = sp.expand(sp.factor(self.G[i, j]))
                if d not in self.G.free_symbols:
                    self.multiplier *= (q - 1)
                    self.G_symbols.remove(d)
            
            # Base case: the trivial group has 1 representation
            if not self.G_symbols:
                assert not self.T.gens # Because we simplified the presentation, (`G` = trivial) => (`T` = trivial)
                return self.multiplier
            
            # Try the lookup table (only when G is constant, i.e. `T` is a point)
            if not self.T.gens:
                v = LOOKUP_TABLE.get(self.G)
                if v:
                    total = self.multiplier * v
                    return total
            
            if FLAG_DEBUG:
                self.display_math('\\text{Considering } G = ' + sp.latex(self.G) + ' \\text{ over } ' + self.T.latex())

            subtotal = 0 # this will be the zeta function before multiplying by self.multiplier

            # Matrix representing elements of H = G / N
            H = self.G[0:n - 1, 0:n - 1]
            # Goal: identify an additive group G_a^r
            # Write      | H | N |
            #        G = +---+---+
            #            | 0 | 1 |
            N = self.G[0:n - 1, n - 1]
            # Writing G = N \rtimes H, we can identify N by taking H = 1
            N_eqs = [ H[i, j] for i in range(n - 1) for j in range(i + 1, n - 1) ]
            N_cases = apply_linear_eqs(self.T, self.G_symbols, N_eqs, N)
            for (T, N, subs) in N_cases:
                N_symbols = list(N.free_symbols.intersection(self.G_symbols))
                r = len(N_symbols) # Note that `r` is not necessarily the rank of `N`, only an upper bound. The actual rank `k` may vary over `T` and is computed below.
                
                # Short-cut for when `N` = trivial
                if r == 0:
                    subtotal += TriangularGroup(T, H, self.depth + 1).zeta_function()
                    continue

                # Apply the substitutions to H
                H_subs = simplify_matrix(T.gens, T.eqs, subs.apply(H))

                # Let M be the (n - 1) x r matrix representing the coefficients of
                # the entries of G[0:n - 1, n - 1] in terms of N_symbols
                M = sp.Matrix([ get_linear_coefficients(N_symbols, x) for x in N ])
                                
                # Write M = A * (I_k & 0 \\ 0 & 0) * B, with A and B invertible, and k the rank of M.
                # Then the first k columns of A are generators of the column space of M
                # This decomposition may depend on certain equations
                for (T, k, A, A_inv) in find_independent_columns(T, M):
                    # If k = 0, we have G = H, so we can just continue with H
                    if k == 0:
                        subtotal += TriangularGroup(T, simplify_matrix(T.gens, T.eqs, H_subs), self.depth + 1).zeta_function()
                        continue
                    # H_eff describes how H acts on the linearly independent columns
                    H_eff = (A_inv * H_subs * A)[0:k, 0:k]
                    # After a permutation of the columns, H_gens should be in upper triangular form
                    H_eff = permute_matrix_to_upper_triangular(H_eff)
                    # Since det(A) need not be 1, there might be denominators in H_eff. We must clear them!
                    # Also, there might be unsimplified fractions in there: we use sp.expand(sp.factor(-)) for this.
                    for i in range(k):
                        for j in range(i, k):                            
                            H_eff[i, j] = sp.expand(sp.factor(H_eff[i, j]))
                            denom = sp.fraction(H_eff[i, j])[1]
                            if denom != 1:
                                H_eff = Substitution({
                                    x: x * denom for x in H_eff[i, j].free_symbols.intersection(self.G_symbols)
                                }).apply(H_eff)
                    
                    # So now, chi ↦ chi * H_eff
                    first_i = max([ 0 ] + [ int(str(x)[4:-1]) + 1 for x in T.gens if x.is_Dummy ]) # TODO: this is a bit hacky..
                    X_gens = [ sp.Dummy(f'x_{{{first_i + i}}}') for i in range(k) ]
                    X = System(X_gens, [], [])
                    chi = sp.Matrix([ X_gens ])
                    
                    if FLAG_DEBUG:
                        im_chi = chi * H_eff
                        self.display_math('\\text{Considering } H = ' + sp.latex(H_subs) + '\\text{ acting on } \\chi = ' + sp.latex(chi) + '\\overset{H}{\\mapsto} ' + sp.latex(im_chi))

                    # Find representatives
                    representatives = find_orbit_stabilizer_representatives(T, self.G_symbols, H_eff, X, chi)
                    for (T, chi, stabilizer, index) in representatives:
                        new_G = sp.Matrix(stabilizer.apply(H_subs)) # make sure this is a copy, so that H_subs doesn't get altered magically!

                        if FLAG_DEBUG:
                            self.display_math('\\bullet \\text{ Representative } \\chi = ' + sp.latex(chi) + ' \\text{ over } ' + T.latex() + '\\text{ has stabilizer } ' + sp.latex(new_G) + ' \\text{ of index } ' + sp.latex(index))

                        part = TriangularGroup(T, new_G, self.depth + 1).zeta_function() * index**(-s)

                        if FLAG_DEBUG:
                            self.display_math('\\Rightarrow ' + sp.latex(part))

                        subtotal += part

            # Store result in lookup table (only if G is constant)
            if not self.T.gens:
                LOOKUP_TABLE.put(self.G, subtotal)

            # Multiply by self.multiplier
            total = self.multiplier * subtotal

            if FLAG_DEBUG:
                self.display_math('\\text{In total, obtain } ' + sp.latex(total))

            return total
        except Exception as e:
            self.display_math('\\text{Error in } G = ' + sp.latex(self.G) + ' \\text{ over } ' + self.T.latex())
            assert False, str(e)
