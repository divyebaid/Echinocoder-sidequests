# Testing the second round canonicalisation (Simplex 1) for violations of bijectivity
import numpy as np
from itertools import permutations
from scipy.optimize import linear_sum_assignment

def generate_permutations(size: int, for_index=False):
    # Returns all (size)! arrays of integers from 1 to (size) by default
    # or all (size)! arrays of integers from 0 to (size-1) if for_index
    perm_range = range(size) if for_index else range(1, size+1)
    all_perms = [list(perm) for perm in permutations(perm_range, size)]
    
    return all_perms


def randomised_cumulative_sum_matrices(initial_permutation: list, only_one_random: bool, debug=False):
    # Generates all possible row-wise randomised matrices cumulatively summing to the given permutation
    # Returns [[row perm, 1s and 0s matrix, cumul sum matrix] for relevant perms], initial perm
    if debug:
        print('---RANDOMISED_CUMUL_SUM_MATRICES_SUBROUTINE---\nInitial perm: \n', initial_permutation)
    matrix_size = len(initial_permutation)
    assert matrix_size > 1  # ignore trivial case n, k = 1
    assert sorted(initial_permutation) == list(range(1, matrix_size+1))  # check that permutation is valid

    initial_ones_matrix = np.zeros(shape=(matrix_size, matrix_size))
    initial_ones_matrix[-1, :] = 1
    for col, num in enumerate(initial_permutation):
        initial_ones_matrix[-num:-1, col] = 1

    if debug:
        print('\nInitial perm ones matrix: \n', initial_ones_matrix)

    if only_one_random:
        randomise_ones_permutations = [np.random.permutation(matrix_size-1).tolist()]
    else:
        randomise_ones_permutations = generate_permutations(matrix_size-1, for_index=True)
    randomised_ones_matrices = []
    randomised_cumulative_sum_matrices = []
    for randomise_perm in randomise_ones_permutations:
        randomise_perm.append(matrix_size-1)
        new_perm_ones_matrix = initial_ones_matrix[randomise_perm, :]
        new_cumulative_sum_matrix = np.cumsum(new_perm_ones_matrix, axis=0)
        randomised_ones_matrices.append([randomise_perm, new_perm_ones_matrix, new_cumulative_sum_matrix])
        randomised_cumulative_sum_matrices.append(new_cumulative_sum_matrix)

    if debug:
        print('\nRandom perm, ones matrix, cumul sum matrix:')
        print(*(item for items in randomised_ones_matrices for item in items), sep='\n')

    return randomised_ones_matrices, initial_permutation


def row_lexicographical_order(n: int, k: int, row: list|np.ndarray, debug=False):
    # Divides row into n sets of k-vectors (rows)
    # Returns sorted row, sort order, initial row 
    if debug:
        print('---LEXICOGRAPHICAL ORDER SUBROUTINE---\nInitial row: \n', row)

    assert len(row) == n*k
    modular_base = np.max(row) + 1

    divided_row = np.array(np.split(np.array(row), n))
    if debug:
        print('\nDivided row: \n', divided_row)
    
    powers = modular_base ** np.arange(k-1,-1,-1)
    modular_sums = divided_row @ powers
    if debug:
        print('\nModular sums: \n', modular_sums)
    
    sort_idxs = np.argsort(modular_sums)
    modular_sums_sorted = modular_sums[sort_idxs]
    divided_row_sorted = divided_row[sort_idxs]
    if debug:
        print('\n Sort order: \n', sort_idxs)
        print('Sorted modular sums: \n', modular_sums_sorted)
        print('Divided row sorted: \n', divided_row_sorted)
    
    row_sorted = divided_row_sorted.flatten()

    return row_sorted, sort_idxs, row


def canonicalisations(initial_matrix: np.ndarray, n: int, k: int, debug=False):
    # Performs first (based on last row, on each k columns) and second (on each k element in each row) canonicalisations
    # Returns canonicalised matrix, first canonicalised matrix, original matrix
    if debug:
        print('---CANONICALISATIONS SUBROUTINE---\nInitial matrix: \n', initial_matrix)

    assert initial_matrix.shape == (n*k, n*k)
    col_groups = np.arange(n*k).reshape(n, k)

    # First canonicalisation
    first_canon_sort_order = row_lexicographical_order(n=n, k=k, row=initial_matrix[-1])[1]
    new_col_groups = col_groups[first_canon_sort_order].flatten()
    first_canon_matrix = initial_matrix[:, new_col_groups]
    if debug:
        print('\nFirst canonicalised matrix: \n', first_canon_matrix)

    # Second canonicalisation
    first_canon_matrix_copy = first_canon_matrix.copy()
    second_canon_matrix = first_canon_matrix.copy()
    for row_num, row in enumerate(first_canon_matrix_copy[:-1]):
        new_row = row_lexicographical_order(n=n, k=k, row=row)[0]
        second_canon_matrix[row_num] = new_row
    if debug:
        print('\nSecond canonicalised matrix: \n', second_canon_matrix)

    return second_canon_matrix, first_canon_matrix, initial_matrix


def decanonicalise_matrix(canon_matrix: np.ndarray, n: int, k: int, debug=False):
    # Attempts to de-(second)canonicalise matrix
    # Returns success bool, de-(second)canonicalised matrix, canonicalised matrix
    if debug:
        print('---DECANONICALISATION---\nCanonicalised matrix: \n', canon_matrix)
    assert canon_matrix.shape == (n*k, n*k)

    decanon_matrix = canon_matrix.copy()
    for idx in range(-2, -n*k-1, -1):
        decanon_matrix_copy = decanon_matrix.copy()
        row = decanon_matrix_copy[idx]
        row_below = decanon_matrix_copy[idx+1]
        row_grouped = row.reshape(n, k)
        row_below_grouped = row_below.reshape(n, k)
        cost_matrix = np.empty((n, n))

        for i in range(n):
            for j in range(n):
                row_group_diff = row_below_grouped[i] - row_grouped[j]
                cost_matrix[i,j] = np.max(row_group_diff) if np.all(row_group_diff >= 0) and np.max(row_group_diff) <= 1 else np.inf
        if debug:
            print('\nCost matrix: \n', cost_matrix)

        opt_row_idx, opt_col_idx = linear_sum_assignment(cost_matrix)
        if debug:
            print('\nOptimal row, col idxs: \n', (opt_row_idx, opt_col_idx))

        if np.isinf(cost_matrix[opt_row_idx, opt_col_idx]).any():
            print(f'WARNING: COULD NOT PERMUTE ROW {idx} TO MAKE DIFFERENCES <= 1')
            return False, decanon_matrix, canon_matrix
        else:
            decanon_row = row_grouped[opt_col_idx].reshape(-1)
            decanon_matrix[idx] = decanon_row
    
    # Checking if solution obeys rule of maximum element-wise difference being 1
    success_bool = True
    diffs_matrix = np.diff(decanon_matrix, axis=0)
    max_diff = np.max(diffs_matrix)
    if max_diff != 1:
        print(f'WARNING: ERROR IN DECANONICALISING, DIFFS MATRIX: \n', diffs_matrix)
        success_bool = False

    return success_bool, decanon_matrix, canon_matrix


if __name__ == '__main__':
    # Testing the inversion of second canonicalisation
    initial_perm = np.random.permutation(np.arange(1,13)).tolist()
    n = 4
    k = 3
    initial_matrix = randomised_cumulative_sum_matrices(initial_perm, True)[0][0][2]
    two_canon_matrix, one_canon_matrix, initial_matrix = canonicalisations(initial_matrix, n, k)
    success_bool, decanon_matrix, canon_matrix = decanonicalise_matrix(two_canon_matrix, n, k, True)
    if success_bool:
        if np.array_equal(decanon_matrix, one_canon_matrix):
            print('GREAT SUCCESS')
        else:
            print('PROBLEM')