from vertex_simplex_architecture_SHA import *
import numpy as np
from EncDec import *
    

class Hash_EncDec():

    def __init__(self, n, k, simplex_map = None):
        self.n = n
        self.k = k
        self.hash = Hash()
        if simplex_map:
            self.simplex_map = simplex_map
        else:
            self.simplex_map = Simplex_Map(n, k)


    def encode(self, data):
        
        lin_comb, offsets = simplex_2_preprocess_steps(data, preserve_scale_in_step_2 = False, canonicalise = True)
        print(lin_comb)
        encoded_data = []
        for mlc in offsets.mlcs():
            encoded_data.append(mlc.coeff)
        code = list(self.hash.lincomb_hash(lin_comb))
        encoded_data += code
        return np.array(encoded_data)
        
    def decode(self, n ,k , data):
        ans = np.zeros(shape = (n,k))
        for i in range(n):
            ans[i] += data[:k]
        print(ans)
        data = data[k:]
        
        deltas, ejis = Simplex_Map(n,k).get_lin_comb(data)
        ejis = [eji.to_array() for eji in ejis]
        raw_lin_comb = zip(deltas, ejis)
        lin_comb = self.fix_lin_comb(raw_lin_comb)
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
            #new_lin_comb[i][0] /= (indices[i]+1)
        
        return new_lin_comb


if __name__ == "__main__":
    
    some_input = np.asarray([[2, 4], [1, 6], [8,3]])
    n = len(some_input)
    k = len(some_input[0])
    s_map = Simplex_Map(n ,k)
    enc_dec = Hash_EncDec(n ,k, simplex_map = s_map)
    output = enc_dec.encode(some_input)
    

    print("Embedding:")
    print(f"{some_input}")
    print("leads to:")
    print(f"{output}")

    print("Decoding then gives:")

    decoded_input = enc_dec.decode(n,k,output)
    print(f"Decoded: {decoded_input}")
