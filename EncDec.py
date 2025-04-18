#!/usr/bin/env python

from collections.abc import Iterable
from fractions import Fraction
from itertools import pairwise
import numpy as np
import tools
from ndenumerate_slice import ndenumerate_fixed_axes_slice

from tools import numpy_array_of_frac_to_str
from tuple_ize import tuple_ize

class MonoLinComb:
    def __init__(self, coeff, basis_vec):
        self.coeff = coeff
        self.basis_vec = basis_vec

    def __repr__(self):
        return f"MonoLinComb({repr(self.coeff)}, {repr(self.basis_vec)})"

    def __eq__(self, other):
        return self.coeff == other.coeff and np.array_equal(self.basis_vec, other.basis_vec)

    def __len__(self):
        return 1 # We are a mono (mono=1) lin comb after all.

class LinComb:
    def __init__(self, initialiser=None):
        self.coeffs = []
        self.basis_vecs = []
        if initialiser is not None:
            self += initialiser

    def mlcs(self): # MLCS = Mono Lin CombinationS
        return (MonoLinComb(c,b) for c,b in zip(self.coeffs, self.basis_vecs))

    def __len__(self):
        assert len(self.coeffs) == len(self.basis_vecs)
        return len(self.coeffs)

    def to_numpy_array(self):
        ans = None
        first = True
        for c,b in zip(self.coeffs, self.basis_vecs):
            if first == True:
                ans = c*np.asarray(b)
                first = False
            else:
                ans += c*np.asarray(b)
        return ans

    def __add__(self, stuff):
        return LinComb((self, stuff))

    def __iadd__(self, stuff):
        #print(f"In iadd see stuff of type {stuff}")
        # Note that __add__ does not automatically consolidate. I.e. (3i+2j) + (5i) becomes (3i+2j+5i) not (8i+2j).
        # It is the user's responsibility to perform consolidation manually if they wish it to happen!!
        if isinstance(stuff, LinComb):
            self.coeffs.extend(stuff.coeffs)
            self.basis_vecs.extend(stuff.basis_vecs)
            return self

        if isinstance(stuff, MonoLinComb):
            self.coeffs.append(stuff.coeff)
            self.basis_vecs.append(stuff.basis_vec)
            return self

        if isinstance(stuff, Iterable):
            for elt in stuff:
                self += elt # Recurse!
            return self
        
        print(f"DDDD {stuff}")
        raise ValueError("LinComb.__iadd__ only knows how to add LimCombs and MonoLinCombs and iterables containing those.")

    def is_consolidated(self):
        # Need to convert 2D numpy arrays to tup(tup()) and 1D numpy arrays to tup() so that they are hashable:
        
        #print(f"Being asked for consolidation state of {self.basis_vecs}")
        basis_vecs_as_tuptups = [ tuple_ize(bv) for bv in self.basis_vecs ]
        #print(f"turned {self.basis_vecs} into {basis_vecs_as_tuptups}")
        return len(set(basis_vecs_as_tuptups)) == len(basis_vecs_as_tuptups)

    def __repr__(self):
        tmp = list(MonoLinComb(c,np.asarray(b)) for c,b in zip(self.coeffs, self.basis_vecs))
        return str(f"LinComb({tmp})")

    def __eq__(self, other):
        if len(self.coeffs) != len(other.coeffs):
            return false

        assert len(self.coeffs) == len(self.basis_vecs)
        assert len(other.coeffs) == len(other.basis_vecs)

        # Note that the order is required to match here, so eq means "same lin com in same order".
        for i,j in zip(self.mlcs(), other.mlcs()):
            if i != j:
               return False

        return True

def array_to_lin_comb(arr: np.array, fixed_axes=None, debug=False):
        lin_comb = LinComb()
        if fixed_axes is None:
            locations = np.ndenumerate(arr)
        else:
            locations = ndenumerate_fixed_axes_slice(arr, fixed_axes=fixed_axes)

        for index, coeff in locations:
            #if debug:
            #    print(f"Considering pos {index} and coeff {coeff}.")
            basis_vec = np.zeros_like(arr)
            basis_vec[index] = 1
            lin_comb += MonoLinComb(coeff, basis_vec)
        return lin_comb

def barycentric_subdivide(lin_comb: LinComb, return_offset_separately=False, preserve_scale=True, debug=False, use_assertion_self_test=False):
    """
        * If preserve_scale is True (default) then the sum of the coeffiencients is preserved. Equivalently, the one
          norm of each basis vector iw preserved at 1 if already at 1.
    """
    
    if debug:
        print(f"lin_comb is\n{lin_comb}")

    if len(lin_comb) == 0:
        return LinComb()

    assert len(lin_comb) >= 1

    assert lin_comb.is_consolidated() # Note that LinComb does not consolidate elements, but our use case should only encounter conslolidated elements anyway! Or so we think. If that is not so, we need to know about it here!

    # Sort by coefficient in linear combination, big -> small
    # We need a list below as we will consume it twice when generating the diff_lin_comb
    sorted_lin_comb = sorted(zip(lin_comb.coeffs, lin_comb.basis_vecs), key=lambda x: x[0], reverse=True)

    if debug:
        print(f"sorted_lin_comb is\n{sorted_lin_comb}")

    coeffs = [ x for x, _ in sorted_lin_comb ]
    basis_vecs = [ x for _ , x in sorted_lin_comb ]

    if preserve_scale:
        diff_lin_comb = LinComb(MonoLinComb((fac := (i+1))*(x-y), sum(basis_vecs[:i+1], start=0*basis_vecs[0]+Fraction())/fac) for i, (x,y) in enumerate(pairwise(coeffs)))
        offset_mono_lin_comb = MonoLinComb((fac := len(basis_vecs))*coeffs[-1], sum(basis_vecs, start=0*basis_vecs[0]+Fraction())/fac)
    else:
        diff_lin_comb = LinComb(MonoLinComb((x-y), sum(basis_vecs[:i+1], start=0*basis_vecs[0])) for i, (x,y) in enumerate(pairwise(coeffs)))
        offset_mono_lin_comb = MonoLinComb(coeffs[-1], sum(basis_vecs, start=0*basis_vecs[0]))

    if debug:
        print(f"diff_lin_comb is\n{diff_lin_comb}")
        print(f"offset_mono_lin_comb is\n{offset_mono_lin_comb}")

    if return_offset_separately:
        ans = diff_lin_comb, offset_mono_lin_comb
    else:
        ans = diff_lin_comb + offset_mono_lin_comb

    """
    No need to do the following, but conceptually useful as documentation as it shows us one of the things
    that this subroute intends to achieve:
    """
    if use_assertion_self_test:
        assert np.allclose( lin_comb.to_numpy_array().astype(float), (diff_lin_comb + offset_mono_lin_comb).to_numpy_array().astype(float))

    if debug:
        print(f"About to return \n{ans}")

    return ans

def simplex_1_preprocess_steps(set_array : np.array, 
                               preserve_scale_in_step_1=False,
                               preserve_scale_in_step_2=True,
                               canonicalise=False,
                               use_assertions=False,
                               debug=False):

    """
    Step 1: 

    Turn the array (which represents a set) into a linear combination of coefficients and Eji basis elements.
    Conceptually this step is turning:

       set_array = [[2,8],[4,5]]

    into

       lin_comb_0 = 2 * [[1,0],[0,0]] + 8 * [[0,1],[0,0]] + 4 * [[0,0],[1,0]] + 5 * [[0,0],[0,1]]

    """

    lin_comb_0 = array_to_lin_comb(set_array)

    if use_assertions:
        assert np.allclose(set_array.astype(float), lin_comb_0.to_numpy_array().astype(float))


    """
    Step 2:

    Re-write lin_comb_0 as a sum of the offset (which is the minimum coefficient times some perm-invariant 
    basis element like (say)[[1,1],[1,1]] and some (so called) differences.  
    The latter are a set of non-negative coefficients times other basis vectors only 
    containing zeros and ones.  Conceptually this step is turning:

       lin_comb_0 = 2 * [[1,0],[0,0]] + 8 * [[0,1],[0,0]] + 4 * [[0,0],[1,0]] + 5 * [[0,0],[0,1]]

    into

       lin_comb_1 + offset

    where the first differences are:
    
        lin_comb_1 = (8-5) * [[0,1],[0,0]]
                   + (5-4) * [[0,1],[0,1]]  
                   + (4-2) * [[0,1],[1,1]] 

    and where the offset is:

        offset = 2 * [[1,1],[1,1]]

    The above example assumed that preserve_scale=False is supplied to barycentric_subdivide, and that thus
    the one-norm of the basis vecs in the linear combination is growing as you go down the list, rather than
    constant as it would be if preserve_scale=True had been used instead.
    """

    lin_comb_1_first_diffs, offset = barycentric_subdivide(lin_comb_0, return_offset_separately=True, preserve_scale=preserve_scale_in_step_1, use_assertion_self_test=True)

    if use_assertions:
        assert np.allclose(set_array.astype(float), (lin_comb_1_first_diffs+offset).to_numpy_array().astype(float))

    """
    Step 3:

    Now we do a barycentric subdivision of lin_comb_1, storing the answer in lin_comb_2.  
    The purpose of this step is to make the resulting basis vectors sufficiently complicated that the
    process of canonicalise them will allow the set of all vertices to retain enough information that the
    canonicalisation process can be undone in all the materially important ways - according to PKH claim..
    [Provably the canonicalisation process would delete information if it were applied to lin_comb_1 directly.] 

    In principle this subdivision should be done with preserve_scale=True (as the vertices of mid-points of 
    simplex edges are things like (v1+v2)/2 not (v1+v2).  However, since this introduces a lot of fractions 
    into the output, a lot of debugging is done with preserve_scale=False.
    """

    lin_comb_2_second_diffs = barycentric_subdivide(lin_comb_1_first_diffs, return_offset_separately=False, preserve_scale=preserve_scale_in_step_2, use_assertion_self_test=True)

    if use_assertions:
        assert np.allclose(set_array.astype(float), (lin_comb_2_second_diffs + offset).to_numpy_array().astype(float))

    if not canonicalise:
        return lin_comb_2_second_diffs, offset


    """
    Step 4:

    Canonicalise the basis vectors.

    We hope this step is a bijection (given the domain).  PKH claims it is, but I am suspicious. Claim is being tested!
    """

    lin_comb_3_canonical = LinComb(( MonoLinComb(coeff, tools.sort_np_array_rows_lexicographically(basis_vec)) for coeff, basis_vec in zip(lin_comb_2_second_diffs.coeffs, lin_comb_2_second_diffs.basis_vecs) ))

    return lin_comb_3_canonical, offset

def simplex_2_preprocess_steps(set_array : np.array, 
                               preserve_scale_in_step_1=False,
                               preserve_scale_in_step_2=True,
                               canonicalise=False,
                               use_assertions=False,
                               debug=False):

    if debug:
        print(f"simplex_2_preprocess_steps was asked to encode {set_array}")

    n,k = set_array.shape

    """
    Step 1: 

    Turn the array (which represents a set) into a linear combination of coefficients and Eji basis elements, separated by component.
    Conceptually this step is turning:

       set_array = [[2,8],[4,5]]

    into

       lin_comb_0[0] = 2 * [[1,0],[0,0]] + 4 * [[0,0],[1,0]]  # "x"-components
       lin_comb_0[1] = 8 * [[0,1],[0,0]] + 5 * [[0,0],[0,1]]  # "y"-components

    """

    lin_comb_0 = [ array_to_lin_comb(set_array, fixed_axes = {1:cpt_index}) for cpt_index in range(k) ]
    if debug:
        print(f"lin_comb_0 was {lin_comb_0}")
    if use_assertions:
        assert np.allclose(set_array.astype(float), LinComb(lin_comb_0).to_numpy_array().astype(float))

    """
    Step 2:

    Re-write each element of lin_comb_0 as a sum of an offset (which is the minimum coefficient times a 
    perm-invariant element like (say) [[1,0],[1,0]] and some (so called) differences.  
    The latter are a set of non-negative coefficients times other basis vectors only 
    containing zeros and ones.  Conceptually this step is turning:

       lin_comb_0[0] = 2 * [[1,0],[0,0]] + 4 * [[0,0],[1,0]]  # "x"-components, and
       lin_comb_0[1] = 8 * [[0,1],[0,0]] + 5 * [[0,0],[0,1]]  # "y"-components

    into

       lin_comb_1 + offset

    where the first differences are:
    
        lin_comb_1  = (4-2) * [[1,0],[0,0]];    offset  = 2 * [[1,0], [1,0]]   # and
        lin_comb_1 += (8-5) * [[0,1],[0,0]];    offset += 2 * [[0,1], [0,1]] 

    The above example assumed that preserve_scale=False is supplied to barycentric_subdivide, and that thus
    the one-norm of the basis vecs in the linear combination is growing as you go down the list, rather than
    constant as it would be if preserve_scale=True had been used instead.
    """
    
    lin_comb_1_first_diffs = [None] * k
    offsets = [None] *k

    assert len(lin_comb_0) == k
    assert len(lin_comb_1_first_diffs) == k
    assert len(offsets) == k

    for i in range(k):
        lin_comb_1_first_diffs[i], offsets[i] = barycentric_subdivide(lin_comb_0[i], return_offset_separately=True, preserve_scale=preserve_scale_in_step_1, use_assertion_self_test=True)
        if debug:
            print(f"lin_comb_1[{i}] was {lin_comb_1_first_diffs[i]}")
            print(f"offsets[{i}] was {offsets[i]}")
            print()

    # Here is the merge phase:
    lin_comb_1_first_diffs = LinComb(lin_comb_1_first_diffs)
    offsets = LinComb(offsets)

    if debug:
        print(f"After the merge phase lin_comb_1 was {lin_comb_1_first_diffs}")
        print(f"offsets was {offsets}")
        print()

    if use_assertions:
        assert np.allclose(set_array.astype(float), (lin_comb_1_first_diffs+offsets).to_numpy_array().astype(float))
        if debug:
            print("Happy after step 2")

    """
    Step 3:

    Now we do a barycentric subdivision of lin_comb_1, storing the answer in lin_comb_2.  
    The purpose of this step is to make the resulting basis vectors sufficiently complicated that the
    process of canonicalise them will allow the set of all vertices to retain enough information that the
    canonicalisation process can be undone in all the materially important ways - according to PKH claim..
    [Provably the canonicalisation process would delete information if it were applied to lin_comb_1 directly.] 

    In principle this subdivision should be done with preserve_scale=True (as the vertices of mid-points of 
    simplex edges are things like (v1+v2)/2 not (v1+v2).  However, since this introduces a lot of fractions 
    into the output, a lot of debugging is done with preserve_scale=False.
    """

    lin_comb_2_second_diffs = barycentric_subdivide(lin_comb_1_first_diffs, return_offset_separately=False, preserve_scale=preserve_scale_in_step_2, use_assertion_self_test=True)

    if use_assertions:
        assert np.allclose(set_array.astype(float), (lin_comb_2_second_diffs + offsets).to_numpy_array().astype(float))
        if debug:
            print("Happy after step 3")

    if not canonicalise:
        return lin_comb_2_second_diffs, offsets


    """
    Step 4:

    Canonicalise the basis vectors.

    We hope this step is a bijection (given the domain).  PKH claims it is, but I am suspicious. Claim is being tested!
    """

    lin_comb_3_canonical = LinComb(( MonoLinComb(coeff, tools.sort_np_array_rows_lexicographically(basis_vec)) for coeff, basis_vec in zip(lin_comb_2_second_diffs.coeffs, lin_comb_2_second_diffs.basis_vecs) ))

    return lin_comb_3_canonical, offsets




    

