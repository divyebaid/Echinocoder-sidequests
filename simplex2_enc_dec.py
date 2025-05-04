from vertex_simplex_architecture import *
import numpy as np
from itertools import pairwise
from collections import namedtuple
from tools import sort_np_array_rows_lexicographically, sort_each_np_array_column
from tools import ascending_data as ascending_data_from_tools
import hashlib
from dataclasses import dataclass, field
from typing import Self
from MultisetEmbedder import MultisetEmbedder
from typing import Any


class Embedder(MultisetEmbedder):

    def __init__(self, n, k, simplex_map=None):
        self.n = n
        self.k = k
        if simplex_map:
            self.simplex_map = simplex_map
        else:
            self.simplex_map = Simplex_Map(n, k)

    def embed_kOne(self, data: np.ndarray, debug=False) -> (np.ndarray, Any):
        metadata = None
        return MultisetEmbedder.embed_kOne_sorting(data), metadata

    def embed_generic(self, data: np.ndarray, debug=False) -> (np.ndarray, Any):
        assert MultisetEmbedder.is_generic_data(data) # Precondition
        if debug:
            print(f"data is {data}")
    
        n,k = data.shape
        assert self.n == n and self.k ==k

        """
        example data is [[ 4  2  3]
                         [-3  5  1]
                         [ 8  9  2]
                         [ 2  7  2]] .
        This is four vectors (n-4) in three dimensions (k=3).
        """

        # The following "ascending data" has the x-components in ascending order, the y-components in asceding order,
        # and so on. This has broken up the vectors.  I.e. the j=1 vector in ascending_data is not likely to
        # be any of the vectors in the input (unless the data was already sorted appropriately).
        # You can think of "ascending data" as representing all the things we want to encode EXCEPT the associations
        # which link every element of each vector up in the right way.
        ascending_data = ascending_data_from_tools(data)

        # We need to extract the smallest x, the smallest y, the smallest z (and so on) as these form some of the
        # outputs of the embedding.
        min_elements = ascending_data[0]
        if debug:
            print("min_elements is ")
            print(min_elements)
            """for our example data min_elements is [-3  2  1]"""

        flattened_data_separated_by_cpt = [ [ ( data[j][i], Eji(j,i) ) for j in range(n) ] for i in range(k) ]
        sorted_data_separated_by_cpt = [sorted(cpt, key = lambda x : -x[0]) for cpt in flattened_data_separated_by_cpt]
        """ for our example data
        sorted_data_separated_by_cpt is 
[(np.int64(8), Eji(j=2, i=0)), (np.int64(4), Eji(j=0, i=0)), (np.int64(2), Eji(j=3, i=0)), (np.int64(-3), Eji(j=1, i=0))]
[(np.int64(9), Eji(j=2, i=1)), (np.int64(7), Eji(j=3, i=1)), (np.int64(5), Eji(j=1, i=1)), (np.int64(2), Eji(j=0, i=1))]
[(np.int64(3), Eji(j=0, i=2)), (np.int64(2), Eji(j=2, i=2)), (np.int64(2), Eji(j=3, i=2)), (np.int64(1), Eji(j=1, i=2))]
        """
        if debug:
            print("sorted_data_separated_by_cpt is ")
            _ = [print(bit) for bit in sorted_data_separated_by_cpt]

        difference_data_by_cpt = [[ (x[0]-y[0], x[1]) for x,y in pairwise(cpt) ] for cpt in sorted_data_separated_by_cpt]
    
        if debug:
            print("difference data is")
            _ = [print(bit) for bit in difference_data_by_cpt]
        """
        difference data is
[(np.int64(4), Eji(j=2, i=0)), (np.int64(2), Eji(j=0, i=0)), (np.int64(5), Eji(j=3, i=0))]
[(np.int64(2), Eji(j=2, i=1)), (np.int64(2), Eji(j=3, i=1)), (np.int64(3), Eji(j=1, i=1))]
[(np.int64(1), Eji(j=0, i=2)), (np.int64(0), Eji(j=2, i=2)), (np.int64(1), Eji(j=3, i=2))]
        """
        difference_data_with_MSVs_by_cpt = [[
            (delta, Maximal_Simplex_Vertex(set([eji for (_, eji) in cpt[0:i + 1]]))) for i, (delta, _) in enumerate(cpt)]
            for cpt in difference_data_by_cpt]
    
        if debug:
            print("difference data with MSVs by cpt:")
            _ = [print(bit) for bit in difference_data_with_MSVs_by_cpt]
        """
        difference data with MSVs by cpt:
        [(np.int64(4), Maximal_Simplex_Vertex(_vertex_set={Eji(j=2, i=0)})),
           (np.int64(2), Maximal_Simplex_Vertex(_vertex_set={Eji(j=2, i=0), Eji(j=0, i=0)})),
             (np.int64(5), Maximal_Simplex_Vertex(_vertex_set={Eji(j=2, i=0), Eji(j=3, i=0), Eji(j=0, i=0)}))]
        [(np.int64(2), Maximal_Simplex_Vertex(_vertex_set={Eji(j=2, i=1)})),
           (np.int64(2), Maximal_Simplex_Vertex(_vertex_set={Eji(j=3, i=1), Eji(j=2, i=1)})),
             (np.int64(3), Maximal_Simplex_Vertex(_vertex_set={Eji(j=3, i=1), Eji(j=1, i=1), Eji(j=2, i=1)}))]
        [(np.int64(1), Maximal_Simplex_Vertex(_vertex_set={Eji(j=0, i=2)})), 
           (np.int64(0), Maximal_Simplex_Vertex(_vertex_set={Eji(j=0, i=2), Eji(j=2, i=2)})),
             (np.int64(1), Maximal_Simplex_Vertex(_vertex_set={Eji(j=0, i=2), Eji(j=3, i=2), Eji(j=2, i=2)}))]
        """

        # Now flatten the difference data:
        difference_data_with_MSVs = [bit for cpt in difference_data_with_MSVs_by_cpt for bit in cpt]
        if debug:
            print("difference data with MSVs:")
            _ = [print(bit) for bit in difference_data_with_MSVs]
            """
            difference data with MSVs:
            (np.int64(4), Maximal_Simplex_Vertex(_vertex_set={Eji(j=2, i=0)}))
            (np.int64(2), Maximal_Simplex_Vertex(_vertex_set={Eji(j=2, i=0), Eji(j=0, i=0)}))
            (np.int64(5), Maximal_Simplex_Vertex(_vertex_set={Eji(j=2, i=0), Eji(j=3, i=0), Eji(j=0, i=0)}))
            (np.int64(2), Maximal_Simplex_Vertex(_vertex_set={Eji(j=2, i=1)}))
            (np.int64(2), Maximal_Simplex_Vertex(_vertex_set={Eji(j=3, i=1), Eji(j=2, i=1)}))
            (np.int64(3), Maximal_Simplex_Vertex(_vertex_set={Eji(j=3, i=1), Eji(j=1, i=1), Eji(j=2, i=1)}))
            (np.int64(1), Maximal_Simplex_Vertex(_vertex_set={Eji(j=0, i=2)}))
            (np.int64(0), Maximal_Simplex_Vertex(_vertex_set={Eji(j=0, i=2), Eji(j=2, i=2)}))
            (np.int64(1), Maximal_Simplex_Vertex(_vertex_set={Eji(j=0, i=2), Eji(j=3, i=2), Eji(j=2, i=2)}))
            """

        sorted_difference_data_with_MSVs = sorted(difference_data_with_MSVs, key=lambda x: -x[0] )
        if debug:
            print("sorted difference data with MSVs:")
            _ = [print(bit) for bit in sorted_difference_data_with_MSVs]
            """
            sorted difference data with MSVs:
            (np.int64(5), Maximal_Simplex_Vertex(_vertex_set={Eji(j=2, i=0), Eji(j=3, i=0), Eji(j=0, i=0)}))
            (np.int64(4), Maximal_Simplex_Vertex(_vertex_set={Eji(j=2, i=0)}))
            (np.int64(3), Maximal_Simplex_Vertex(_vertex_set={Eji(j=3, i=1), Eji(j=1, i=1), Eji(j=2, i=1)}))
            (np.int64(2), Maximal_Simplex_Vertex(_vertex_set={Eji(j=2, i=0), Eji(j=0, i=0)}))
            (np.int64(2), Maximal_Simplex_Vertex(_vertex_set={Eji(j=2, i=1)}))
            (np.int64(2), Maximal_Simplex_Vertex(_vertex_set={Eji(j=3, i=1), Eji(j=2, i=1)}))
            (np.int64(1), Maximal_Simplex_Vertex(_vertex_set={Eji(j=0, i=2)}))
            (np.int64(1), Maximal_Simplex_Vertex(_vertex_set={Eji(j=0, i=2), Eji(j=3, i=2), Eji(j=2, i=2)}))
            (np.int64(0), Maximal_Simplex_Vertex(_vertex_set={Eji(j=0, i=2), Eji(j=2, i=2)}))
            """

        # Barycentrically subdivide:
        deltas_in_current_order = [delta for delta, _ in sorted_difference_data_with_MSVs]
        msvs_in_current_order = [msv for _,msv in sorted_difference_data_with_MSVs]
    
        expected_number_of_vertices = n * k - k
        assert len(deltas_in_current_order) == expected_number_of_vertices
        assert len(msvs_in_current_order) == expected_number_of_vertices
    
        # The coordinates in the barycentric subdivided daughter simplex are differences of the current deltas,
        # which are up-weighted by a linear factor to (1) preserve their sum so that (2) normalised barycentric
        # coordinates transform into identically normalised barycentric coordinates, and so (3) this makes each
        # component approximately identically distributed.
        difference_data_in_subdivided_simplex = [ (  (i+1)*(deltas_in_current_order[i]-
                 (deltas_in_current_order[i+1] if i+1<expected_number_of_vertices else 0)),
                            Eji_LinComb(n, k, msvs_in_current_order[:i+1])) for i in range(expected_number_of_vertices)]

        if debug:
            print("difference data in Barycentrically subdivided simplex:")
            _ = [print(bit) for bit in difference_data_in_subdivided_simplex]
    
        canonical_difference_data = [(delta, msv.get_canonical_form()) for (delta, msv) in difference_data_in_subdivided_simplex]
        if debug:
            print("canonical difference data is:")
            _ = [print(bit) for bit in canonical_difference_data]
    
        assert n*k - k == expected_number_of_vertices
        bigN = 2*(n - 1)*k # Size of the space into which the simplices are embedded.
        # bigN does not count any min elements, which would be extra.
        difference_point_pairs = [(delta, Vertex(eji_lin_com).value(n,k, simplex_map=self.simplex_map)) for (delta, eji_lin_com) in canonical_difference_data]
        if debug:
            print("difference point pairs are:")
            _ = [print(bit) for bit in difference_point_pairs]
    
        second_part_of_embedding = sum([delta * point for delta, point in difference_point_pairs]) + np.zeros(bigN)
        if debug:
            print(f"second bit of embedding is: {second_part_of_embedding}")
    
        # Create a vector to contain the embedding:
        length_of_embedding = self.size_from_n_k(n,k)

        """
        assert length_of_embedding == bigN + k
        assert bigN == 2*(n - 1)*k + 1
        assert length_of_embedding == 2*n*k + 1 - k  # bigN + 2
        assert len(min_elements) == k
        """
        embedding = np.zeros(length_of_embedding, dtype=np.float64)
    
        # Populate the first part of the embedding with the smallest elements of the initial data.
        embedding[:k] = min_elements
        # Populate other half of the embedding:
        embedding[k:bigN + k] = second_part_of_embedding

        if debug:
            print(f"embedding is {embedding}")
            print(f"embedding has length {length_of_embedding}")
    
        metadata = { "ascending_data" : ascending_data, "input_data" : data, }
        return embedding, metadata
    
    def size_from_n_k_generic(self, n: int, k: int) -> int:
        return 2*n*k + 1 - k

def tost(): # Renamed from test -> tost to avoid pycharm mis-detecting / mis-running unit tests!
    calculated = np.array([2, 3, 4, 1, 0])
    expected = np.array([2, 3, 4, 1, 0])
    np.testing.assert_array_equal(calculated, expected)

def run_unit_tests():
    tost() # Renamed from test -> tost to avoid pycharm mis-detecting / mis-running unit tests!

class Decoder():                

    def __init__(self, n, k, simplex_map = None):
        self.n = n
        self.k = k
        if simplex_map:
            self.simplex_map = simplex_map
        else:
            self.simplex_map = Simplex_Map(n, k)
    
    def decode(self, data):
        n = self.n
        k = self.k
        ans = np.zeros(shape = (n,k))
        for i in range(n):
            ans[i] += data[:k]
        print(ans)
        data = data[k:]
        
        deltas, ejis = self.extract_data(data)
        new_ejis = [eji.to_array() for eji in ejis]
        print(new_ejis)
        raw_lin_comb = zip(deltas, new_ejis)
        lin_comb = self.fix_lin_comb(raw_lin_comb)
        print(lin_comb)
        for i in range(len(lin_comb)):
            ans += lin_comb[i][0]*lin_comb[i][1]      
        return ans

#takes linear combination, sorts and permutes the vertices to the way they were originally generated, also normalizes deltas
    def fix_lin_comb(self, unfixed_lin_comb):
        lin_comb = sorted(unfixed_lin_comb, key = lambda x: sum(x[1].flatten()))
        for i in range(len(lin_comb)):
            lin_comb[i] = list(lin_comb[i])
        n = len(lin_comb[0][1])
        k = len(lin_comb[0][1][0])
        new_lin_comb = []
        indices =[]
        for i in range(0, (n-1)*k):
            if lin_comb[i][0] != 0:
                new_lin_comb.append(lin_comb[i])
                indices.append(i)
        print(new_lin_comb)
        print(indices)
        for i in range(1, len(new_lin_comb)):
            benchmark = new_lin_comb[i-1][1]
            fixed_array = np.zeros(shape=(n, k))
            perms = list(permutations(new_lin_comb[i][1]))
            arrays = [np.stack(p).astype(np.int16) for p in perms]
            diffs = [np.linalg.norm(array-benchmark) for array in arrays]
            min_diff = diffs.index(min(diffs))
            fixed_array = arrays[min_diff]
            new_lin_comb[i][1] = fixed_array
            new_lin_comb[i][0] /= (indices[i]+1)
        
        return new_lin_comb

    def extract_data(self, data):
        n = self.n
        k = self.k
        simplex_map = self.simplex_map
        deltas = []
        ejis= []
        for i in range((n-1)*k):
            deltas.append(np.hypot(data[2*i], data[2*i+1]))
            temp = simplex_map.vertex_key[i]
            full = len(temp)
            order = int(np.round((np.arctan2(data[2*i+1], data[2*i]))*full/2/np.pi))
            if order < 0:
                order += full
            print(f"index {i}: {order}")
            ejis.append(temp[order])
        print(deltas)
        print(ejis)
        return deltas, ejis

if __name__ == "__main__":
    
    some_input = np.asarray([[2, 4, 3], [1, 6, 2], [8,3,5]])
    n = len(some_input)
    k = len(some_input[0])
    s_map = Simplex_Map(n ,k)
    embedder = Embedder(n, k, simplex_map = s_map)
    output = embedder.embed(some_input, debug = True)

    print("Embedding:")
    print(f"{some_input}")
    print("leads to:")
    print(f"{output}")

    print("Decoding then gives:")

    decoder = Decoder(n, k, simplex_map = s_map)
    decoded_input = decoder.decode(output[0])
    print(f"Decoded: {decoded_input}")
