import numpy as np

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


#returns powerset without the initial empty tupel
def powerset(s: list):
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

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

Eji = namedtuple("Eji", ["j", "i"])

def eji_set_to_np_array(eji_set, n, k):
    ans = np.zeros(shape=(n, k))
    for (j, i) in eji_set:
        ans[j][i] = 1
    return ans

def eji_set_array_to_point_in_unit_hypercube(eji_set_array, dimension):
    m = hashlib.md5()
    m.update(eji_set_array)
    ans = []
    for i in range(dimension):
        m.update(i.to_bytes())
        real_1, _ = hash_to_64_bit_reals_in_unit_interval(m) # TODO: make use of real_2 as well to save CPU
        ans.append(real_1)
    return np.asarray(ans)

@dataclass
class Maximal_Simplex_Vertex:
    _vertex_set: set[Eji] = field(default_factory=set)

    def __len__(self) -> int:
        return len(self._vertex_set)

    def __iter__(self):
        return iter(self._vertex_set)

    def get_canonical_form(self):
        """Mod out by Sn for this single vertex, ignoring any others."""
        # Method: sort the Eji by the i index, then populate the j's in order.
        sorted_eji_list = sorted(list(self._vertex_set), key=lambda eji: eji.i)
        renumbered_eji_list = [ Eji(j=j, i=eji.i) for j,eji in enumerate(sorted_eji_list)]
        return Maximal_Simplex_Vertex(set(renumbered_eji_list))

    def check_valid(self):
        # every j index in the set must appear at most once
        j_vals = { eji.j for eji in self._vertex_set }
        assert len(j_vals) == len(self._vertex_set)

    def get_permuted_by(self, perm):
        return Maximal_Simplex_Vertex({Eji(perm[eji.j], eji.i) for eji in self._vertex_set})

@dataclass
class Eji_LinComb:
    INT_TYPE = np.uint16 # uint16 should be enough as the eij_counts will not exceed n*k which can therefore reach 65535

    _index : INT_TYPE
    _eji_counts : np.ndarray

    def index(self) -> INT_TYPE:
        """How many things were added together to make this Linear Combination."""
        return self._index

    def __init__(self, n: int, k: int, list_of_Maximal_Simplex_Vertices: list[Maximal_Simplex_Vertex] | None = None):
        self._index = Eji_LinComb.INT_TYPE(0)
        self._eji_counts = np.zeros((n, k), dtype=Eji_LinComb.INT_TYPE, order='C')
        if list_of_Maximal_Simplex_Vertices:
            for msv in list_of_Maximal_Simplex_Vertices: self.add(msv)

    def _setup_debug(self, index: int, eji_counts: np.ndarray): # Really just for unit tests. Don't use in main alg code.
        self._index = Eji_LinComb.INT_TYPE(index)
        self._eji_counts = np.asarray(eji_counts, dtype=Eji_LinComb.INT_TYPE, order='C')

    def add(self, msv: Maximal_Simplex_Vertex):
        self._index += 1
        for j, i in msv: self._eji_counts[j, i] += 1

    def __eq__(self, other: Self):
        return self._index == other._index and np.array_equal(self._eji_counts, other._eji_counts)

    def __ne__(self, other: Self):
        return not self.__eq__(other)

    def get_canonical_form(self) -> Self:
        ans = Eji_LinComb.__new__(Eji_LinComb)
        ans._index = self._index
        ans._eji_counts = sort_np_array_rows_lexicographically(self._eji_counts)
        return ans

    """
    def hash_to_point_in_unit_hypercube(self, n, k):
        m = hashlib.md5()
        m.update(self._eji_counts)
        #print("self._index is")
        #print(self._index)
        # self._index.nbytes returns the number of bytes in self._index as self._index is of a numpy type which provides this
        m.update(np.array([self._index])) # creating an array with a single element is a kludge to work around difficulties of using to_bytes on np_integers of unknown size
        ans = []
        for i in range(dimension):
            m.update(i.to_bytes(8))  # TODO: This 8 says 8 byte integers
            real_1, _ = hash_to_64_bit_reals_in_unit_interval(m)  # TODO: make use of real_2 as well to save CPU
            ans.append(real_1)
        return np.asarray(ans)
"""

#essentially same class as eji_LinComb with some added functionality (mainly summing with other vertices and a function to translate an array to a vertex with that array as eji_counts
class Vertex(Eji_LinComb):

    def __init__(self, *args):
        if len(args)==1:
            eji = args[0]
            assert type(eji) == Eji_LinComb
            self._index = eji._index
            self._eji_counts = eji._eji_counts
        else:
            super().__init__(*args)

    #Note that adding does not commute with other class operations, in particular not with value or to_array
    #e.g. a.value(dim)+b.value(dim) != (a+b).value(dim)
    def __add__(self, other):
        ans = Vertex.__new__(Vertex)
        ans._index = self._index+other._index
        ans._eji_counts = self._eji_counts+other._eji_counts
        return ans
    
    def value(self, n, k, simplex_map=None):
        code = np.zeros(2*(n-1)*k)

        order, full = self.el(n, k, simplex_map=simplex_map)


        theta = order*2*np.pi/full

        i = self.index()-1
        print(f"Value call: index: {i}; order: {order}")
        code[2*i] = np.cos(theta)
        code[2*i+1] = np.sin(theta)
        return code

    def to_array(self):
        return self.get_canonical_form()._eji_counts

    def get_canonical_form(self):
        ans = Vertex.__new__(Vertex)
        ans._index = self._index
        ans._eji_counts = sort_np_array_rows_lexicographically(self._eji_counts)
        return ans

    def array_to_vertex(ejis: np.array):
        ans = Vertex.__new__(Vertex)
        ans._index = 1
        ans._eji_counts = ejis
        return ans

    def el(self, n, k, simplex_map=None):
        if not simplex_map:
            simplex_map = Simplex_Map(n, k)
        vlist = simplex_map.vertex_key[self.index()-1]
        order = vlist.index(self.get_canonical_form())
        full = len(vlist)
        return order, full
        

#Class to hold groups (lists) of vertices and compare them + geometric functionality
class Simplex():
    
    def __init__(self, n: int, k: int, vlist: list[Vertex]):
        self.n = n
        self.k = k
        self.dim = 2*(n-1)*k+1
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
        return sum(self.vlist, start=Vertex(self.n, self.k))
    
    #uses recursive_step function (see beginning of code)
    def barycentric_subdivision(self):
        barycentre = sum(self.vlist, start=Vertex(self.n, self.k))
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
                new_vertex = sum([self.vlist[i] for i in filter(lambda i: ele[i], range(self.num))], start = Vertex(self.n,self.k))
                temp_vlist.append(new_vertex)
            simplex_list.append(Simplex(self.n, self.k, temp_vlist))
        return simplex_list
    

    #calculates projection of a point in the 2(n-1)k+1 dimensional space onto a given subspaces/simplex with vertices v_i by using the condition that at for point p point and projected point p_0 = lamda_i*v_i, (p-p_0) dot v_i = 0 <-> p dot v_i = lamda_j*(v_j dot v_i)
    def projected_point(self, p: np.ndarray):
        assert len(p) == self.dim
        n = self.num
        b = np.zeros(n)
        A = np.zeros(shape=(n,n))
        for i in range(n):
            b[i] = np.dot(p, self.vlist[i].value(self.dim))
            for j in range(n):
                A[i,j] = np.dot(self.vlist[i].value(self.dim), self.vlist[j].value(self.dim))
        assert np.linalg.det(A) != 0
        lamda  = np.linalg.solve(A, b)
        projection = np.zeros(self.dim)
        for i in range(n):
            projection += lamda[i]*self.vlist[i].value(self.dim)
        return projection, lamda, self.vlist

    def distance_to_point(self, p: np.ndarray):
        projection = self.projected_point(p)[0]
        return np.linalg.norm(p-projection, ord=2)

#Code in red comments in the following class is the original code with exploding runtime, actual code runs fast but has not yet been thoroughly checked
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

        """
        self.slist = self.remove_equivalent_simplices(n, k, simplex_list)
        
        #print("Post_reduction length: ", len(self.slist))
        """
        self.vertex_key = self.vertex_index_list()
        for i in range(len(self.vertex_key)):
            print(f"# of v's with index {i}: {len(self.vertex_key[i])}")
            with open(f"vertices_with_index_{i}.txt", "w") as file:
                for vertex in self.vertex_key[i]:
                    file.write(str(vertex))
                    file.write("\n")
                    file.write("=========")
                    file.write("\n")

        

    #generates a vertex from an eji_list in 1 k-dimension
    def list_to_vertex(self, lt: list, n, k, kmax):
        assert len(lt) == n
        vertex = Vertex(n,kmax,[])
        for l in range (n):
            if(lt[l] != 0):
                vertex.add(Maximal_Simplex_Vertex({Eji(j=l,i=k)}))
                #index is hard coded to 1 here so that the index of final vertices in the full dimension matches the ones generated in the encoder -> otherwise hash function will generate different vectors in hypercube
                vertex._index = 1
        return vertex

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
                vlist.append(self.list_to_vertex(v, n, k, kmax))
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
        lin_comb = simplex.projected_point(p)[1]
        print("Projection is: ", simplex.projected_point(p)[0])
        return lin_comb, simplex.vlist   

    def vertex_index_list(self):
        key = [[] for i in range((self.n-1)*self.k)]
        for simplex in self.slist:
            for vertex in simplex.vlist:
                if vertex.get_canonical_form() not in key[vertex._index-1]:
                    key[vertex._index-1].append(vertex.get_canonical_form())
        return key

            #previous limiting factor on the runtime was during this function applied to list of barycentric subdivisions - no longer needed 
"""
    def remove_equivalent_simplices(self, n ,k, simplex_list):
        reduced_list = []
        index_list=[]
        
        for simplex in simplex_list:
            if simplex not in reduced_list:
                reduced_list.append(simplex.get_canonical_form())
        for i in range(len(simplex_list)):
            simplex = simplex_list[i]
            if simplex not in reduced_list:
                reduced_list.append(simplex.get_canonical_form())
                index_list.append(i)
        #with open("unique_indices.txt", "w") as file:
            #for index in index_list:
                #file.write(f"{index}\n")
        return reduced_list
""" 

def recursive_step_2(base_list, v_list, depth):
    if depth ==


class Simplex_Map_2():

    def __init__(self, n ,k):
        self.n = n
        self.k = k
        self.base_vertices = self.vertices_across_all_k()
        self.vertex_key = []
        for i in range(k*(n-1)):
            self.vertex_key.append([])
        self.vertex_list()

    def list_to_vertex(self, lt: list, k_0):
        assert len(lt) == self.n
        vertex = Vertex(self.n,self.k,[])
        for l in range (self.n):
            if(lt[l] != 0):
                vertex.add(Maximal_Simplex_Vertex({Eji(j=l,i=k_0)}))
                #index is hard coded to 1 here so that the index of final vertices in the full dimension matches the ones generated in the encoder -> otherwise hash function will generate different vectors in hypercube
                vertex._index = 1
        return vertex


    def vertices_for_single_k(self, k_0):
        super_list =[]
        a = np.zeros(self.n)
        #if k_0 ==0:
         #   for i in range(1,self.n):
          #      a[-i] += 1
           #     v_list = [self.list_to_vertex(list(a),0)]
           #     super_list.append(v_list)
            #return super_list
        for i in range(self.n-1):
            a[i] += 1
            v_list=[]
            for perm in distinct_permutations(a):
                v_list.append(self.list_to_vertex(list(perm), k_0))
            super_list.append(v_list)
        return super_list

    def vertices_across_all_k(self):
        vlist = []
        for i in range(self.k):
            for x in self.vertices_for_single_k(i):
                vlist.append(x)
        return vlist

    def recursive_step_2(self, vertex, remainder, depth):
        if depth == k*(n-1):
            for v in *remainder: 
                self.vertex_key[depth-1].append(vertex+v)
            return
        else:
            for i in range(len(remainder)):
                new_remainder = remainder.copy()
                sublist = remainder.pop(i)
                for v in sublist:
                    if v = None:
                        continue
                    new_vertex = vertex+v
                    self.vertex_key[depth-1].append(new_vertex)
                    for i in range(len(new_remainder)):
                        for j in range(len(new_remainder[i])):
                            test = new_remainder[i,j]
                            if test is not None and self.is_not_valid_combination(new_vertex, test):
                                new_remainder[i,j] = None
                    self.recursive_step_2(new_vertex, new_remainder, depth+1)
            return
                    
                

    
    
    def combine_to_indices(self, index):
        base_vs = self.base_vertices
        first_index_max = self.k
        second_index_max = self.n-2
        multi_index = [0,0]
        
        return v_list

    def vertex_list(self):
        vlist = []
        return vlist
        
        

if __name__ == "__main__":
    n= 3
    k=3
    simplex_map = Simplex_Map_2(n,k)
    vlist = simplex_map.vertices_across_all_k()
    count = 0
    for x in vlist:
        for y in x:
            count +=1
    print(count)
    print(vlist)