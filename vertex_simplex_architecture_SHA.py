import numpy as np

from EncDec import *

from tools import sort_np_array_rows_lexicographically, sort_each_np_array_column
from tools import ascending_data as ascending_data_from_tools

from distinct_permutations import distinct_permutations

from itertools import chain, combinations, product, permutations
from itertools import pairwise

from collections import namedtuple

import hashlib

from dataclasses import dataclass, field

from typing import Self
from typing import Any

from math import factorial

from copy import deepcopy

#returns powerset without the initial empty tupel
def powerset(s: list):
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def find_combinations(n, max_values, index=0, current=None):
    if current is None:
        current = []
    if index == len(max_values):
        if n == 0:
            yield list(current)
        return
    for i in range(int(np.round(min(max_values[index], n) + 1))):
        current.append(i)
        yield from find_combinations(n - i, max_values, index + 1, current)
        current.pop()

#Function used to generate vertices according to the rule that pre-existing ones in the eji_lincomb must be preserved
#This is essentially barycentric subdivision:
#For a simplex with vertices in a double nested seed list e.g. of form [[[1,0 ,0]], [[0,1,0]], [[0,0,1]]]
#After all recursions will then return the simplices on the barycentric subdivision that can be generated from this
#(the actual barycentre e.g. [1,1,1] is omitted as it is the same for all simplices
#In this example return will be: [ [[1,0,0],[1,1,0]], [[1,0,0],[1,0,1]], [[0,1,0],[1,1,0]], [[0,1,0],[0,1,1]], [[0,0,1],[1,0,1]], [[0,0,1], [0,1,1]] ]
def recursive_step(l: list):
    new_l = l.copy()
    original_l = len(new_l)
    for i in range(original_l):
        if new_l[i][-1].count(0) == 1:
            return new_l
        else:
            for j in range(len(new_l[i][-1])):
                if(new_l[i][-1][j] == 0):
                    copy = new_l[i][-1].copy()
                    copy[j] += 1
                    edit = new_l[i].copy()
                    edit.append(copy)
                    new_l.append(edit)
    for i in range(original_l):
        new_l.pop(0)
    return recursive_step(new_l)

class Hash():
    
    def hash_to_64_bit_reals_in_unit_interval(self, md5):
        x = int.from_bytes(md5.digest(), 'big')
        bot_64_bits = x & 0xffFFffFFffFFffFF
        top_64_bits = x >> 64
        return np.float64(top_64_bits)/(1 << 64), np.float64(bot_64_bits)/(1 << 64)

    def single_hash(self, dim, v):
        dimension = dim
        index, array = v
        m = hashlib.md5()
        m.update(array)
        #print("self._index is")
        #print(self._index)
        # self._index.nbytes returns the number of bytes in self._index as self._index is of a numpy type which provides this
        m.update(np.array([index])) # creating an array with a single element is a kludge to work around difficulties of using to_bytes on np_integers of unknown size
        ans = []
        for i in range(dimension):
            m.update(i.to_bytes(8))  # TODO: This 8 says 8 byte integers
            real_1, _ = self.hash_to_64_bit_reals_in_unit_interval(m)  # TODO: make use of real_2 as well to save CPU
            ans.append(real_1)
        return np.asarray(ans)
    
    def lincomb_hash(self, lincomb):
        n = len(lincomb.basis_vecs[0])
        k = len(lincomb.basis_vecs[0][0])
        dim = 2*(n-1)*k-k+1
        point = np.zeros(dim)
        index = 1
        for mlc in lincomb.mlcs():
            v = (index, np.array(mlc.basis_vec, dtype=np.int16))
            print(f"LinComb index {index}: v : {v}; point: {self.single_hash(dim, v)}")
            point += mlc.coeff*self.single_hash(dim, v)
            index += 1
        return point

    def vertex_hash(self, vertex):
        n, k = vertex.array.shape
        dim = 2*(n-1)*k-k+1
        v= (vertex.index, vertex.array)
        point = self.single_hash(dim, v)
        return point

#essentially same class as eji_LinComb with some added functionality (mainly summing with other vertices and a function to translate an array to a vertex with that array as eji_counts
class Vertex():

    def __init__(self, n, k_0, kmax, lt: list):
        if lt == []:
            self.index=0
            self.array = np.zeros(shape=(n,kmax), dtype=np.int16)
        else:
            assert len(lt) == n
        
            self.index = 1
            self.array = np.zeros(shape=(n,kmax), dtype=np.int16)
            self.array[:,k_0] = np.array(lt, dtype=np.int16)
    
    #Note that adding does not commute with other class operations, in particular not with value or to_array
    #e.g. a.value(dim)+b.value(dim) != (a+b).value(dim)
    def __add__(self, other):
        ans = Vertex.__new__(Vertex)
        ans.index = self.index+other.index
        ans.array = self.array+other.array
        return ans

    def value(self):
        return Hash().vertex_hash(self.get_canonical_form())

    def to_array(self):
        return self.get_canonical_form().array

    def get_canonical_form(self):
        ans = Vertex.__new__(Vertex)
        ans.index = self.index
        ans.array = sort_np_array_rows_lexicographically(self.array)
        return ans

    def array_to_vertex(ejis: np.array):
        ans = Vertex.__new__(Vertex)
        ans.index = 1
        ans.array = ejis
        return ans

    #for print statements during debugging:
    def __repr__(self):
        return f"*Vertex with index {self.index}; {self.array} *"

#Class to hold groups (lists) of vertices and compare them + geometric functionality
class Simplex():
    
    def __init__(self, n: int, k: int, vlist: list[Vertex]):
        self.n = n
        self.k = k
        self.dim = 2*(n-1)*k+1-k
        self.vlist = vlist
        self.num = len(vlist)

    #compares canonicalisations of simplices - technically no longer needed as only used in SimplexMap.remove_equivalent_simplices()
    def __eq__(self, other):
        if (self.n != other.n) or (self.k!=other.k) or (self.num!=other.num): return False
        array_list_self = []
        array_list_other = []
        for i in range(self.num):
            array_list_self.append(self.vlist[i].to_array())
            array_list_other.append(other.vlist[i].to_array())
        a = sorted(array_list_self, key = lambda x: tuple(x.flatten()))
        b = sorted(array_list_other, key = lambda x: tuple(x.flatten()))
        return all(np.array_equal(x, y) for x,y in zip(a, b))

    #combines list of two simplices -> used when an (n-1)-dimensional simplex for a single value of k is combined with the others to form an (n-1)*k-dimensional simplex
    def __add__(self, other):
        assert(self.n == other.n and self.k == other.k)
        temp = self.vlist.copy()
        return Simplex(self.n , self.k, self.vlist+other.vlist)

    #returns Simplex generated by canonicalised vertices
    def get_canonical_form(self):
        new_vertices = [vertex.get_canonical_form() for vertex in self.vlist]
        return Simplex(self.n, self.k, new_vertices)

    def barycentre(self):
        return sum(self.vlist, start=Vertex(self.n, 0, self.k, []))
    
    #uses recursive_step function (see beginning of code)
    def barycentric_subdivision(self):
        barycentre = sum(self.vlist, start=Vertex(self.n,0, self.k,[]))
        simplex_list =[]
        seed_list = []
        a = np.zeros(self.num)
        a[0] += 1
        for perm in distinct_permutations(a):
            seed_list.insert(0, [list(perm)])
        allowed_vertex_combos = recursive_step(seed_list)
        for combo in allowed_vertex_combos:
            temp_vlist = [barycentre]
            for ele in combo:
                new_vertex = sum([self.vlist[i] for i in filter(lambda i: ele[i], range(self.num))], start = Vertex(self.n,0,self.k,[]))
                temp_vlist.append(new_vertex)
            simplex_list.append(Simplex(self.n, self.k, temp_vlist))
        return simplex_list
    

    #calculates projection of a point in the 2(n-1)k+1-k dimensional space onto a given subspaces/simplex with vertices v_i by using the condition that at for point p point and projected point p_0 = lamda_i*v_i, (p-p_0) dot v_i = 0 <-> p dot v_i = lamda_j*(v_j dot v_i)
    def projected_point(self, p: np.ndarray):
        assert len(p) == self.dim
        n = self.num
        b = np.zeros(n)
        A = np.zeros(shape=(n,n))
        for i in range(n):
            b[i] = np.dot(p, self.vlist[i].value())
            for j in range(n):
                A[i,j] = np.dot(self.vlist[i].value(), self.vlist[j].value())
        assert np.linalg.det(A) != 0
        lamda  = np.linalg.solve(A, b)
        projection = np.zeros(self.dim)
        for i in range(n):
            projection += lamda[i]*self.vlist[i].value()
        return projection, lamda, self.vlist

    def distance_to_point(self, p: np.ndarray):
        projection = self.projected_point(p)[0]
        return np.linalg.norm(p-projection, ord=2)


class Simplex_Map():

    #There is no known use for setting subdivided=False, but kept as a toggle just in case
    def __init__(self, n, k, subdivided = True):
        self.n = n
        self.k = k
        
        temp_list = self.simplices_across_all_k(self.n, self.k)
        
        print("Post_cutoff length: ", len(temp_list))
        if subdivided:
            simplex_list = []
            for simplex in temp_list:
                simplex_list += simplex.barycentric_subdivision()
        else:
            simplex_list = temp_list
        
        print("Post_subdivision length: ", len(simplex_list))

        self.slist = simplex_list



    #generates simplices inside 1 k-dimension
    def generate_simplices_for_single_k(self, n, k, kmax):
        seed_list = []
        a = np.zeros(n)
        a[0] += 1
        for perm in distinct_permutations(a):
            seed_list.insert(0, [list(perm)])
        temp_list = recursive_step(seed_list)
        simplex_list = []
        for ele in temp_list:
            vlist = []
            for v in ele:
                vlist.append(Vertex(n, k, kmax, v))
            simplex_list.append(Simplex(n, kmax, vlist))
        return simplex_list

    #It should be possible to introduce the cutoff earlier in this function, if not in the first then certainly in the second for-loop
    #However generating the non-subdivided simplices is entirely negligible compared to other operations, and the small amount of runtime that could be saved is probably not worth the risk of other bugs arising from interactions with the unpacking or the product() method
    #POST: returns the list of all unique (i.e. non equivalent after canonicalisation) simplices before subdivision
    def simplices_across_all_k(self, n, k):
        superlist = []
        simplex_list = []
        cutoff = factorial(n)**(k-1)
        
        for i in range(k):
            superlist.append(self.generate_simplices_for_single_k(n, i, k))
        for simplex_comb in product(*superlist):
            simplex_list.append(sum(simplex_comb, start=Simplex(n,k, [])))
        
        return simplex_list[:cutoff]

        #iterates through all simplices, calcs distance to point and returns simplex that is closest to point  
    def choose_simplex(self, p: np.ndarray):
        dist_list = []
        simplex_list = self.slist
        print("Default: ", simplex_list[0].vlist)
        dist_list = [x.distance_to_point(p) for x in simplex_list]
        print(min(dist_list))
        chosen_simplex = simplex_list[min(enumerate(dist_list), key = lambda x: x[1])[0]]
        print("Chosen simplex: ", chosen_simplex.vlist)
        return chosen_simplex

        #returns the lamda_i and vertices (that were hashed to the hypercube) of the projected_point (within error of the actual point)
    def get_lin_comb(self, p: np.ndarray):
        simplex = self.choose_simplex(p)
        simplex.vlist = sorted(simplex.vlist, key=lambda x:x.index)
        for i in range(len(simplex.vlist)):
            simplex.vlist[i] = simplex.vlist[i].get_canonical_form()
        lin_comb = simplex.projected_point(p)[1]
        print("Projection is: ", simplex.projected_point(p)[0])
        return lin_comb, simplex.vlist   
    