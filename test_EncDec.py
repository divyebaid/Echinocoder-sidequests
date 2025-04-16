from fractions import Fraction
from itertools import pairwise
import numpy as np

from EncDec import array_to_lin_comb
from EncDec import barycentric_subdivide
from EncDec import LinComb, MonoLinComb
#from EncDec import Chain
from EncDec import pretty_print_lin_comb
from tools import numpy_array_of_frac_to_str

def test_BarycentricSubdivide_split_not_preserving_scale():
    print("###########################################")

    pIn = LinComb([MonoLinComb(-1, np.array([1, 0, 0])),
                   MonoLinComb(-7, np.array([0, 1, 0])),
                   MonoLinComb(10, np.array([0, 0, 1]))])

    pOut, qOut = barycentric_subdivide(pIn, return_offset_separately=True, preserve_scale=False, debug=True)

    print(f"pOut\n{pOut}\nqOut{qOut}")

    pOut_expected = LinComb([MonoLinComb(11, [0, 0, 1]),
                             MonoLinComb(6, [1, 0, 1])])
    qOut_expected = MonoLinComb(-7, [1, 1, 1])

    assert pOut == pOut_expected
    assert qOut == qOut_expected

def test_BarycentricSubdivide_no_split_not_preserving_scale():
    print("###########################################")

    pIn = LinComb([MonoLinComb(-1, np.array([1, 0, 0])),
                   MonoLinComb(-7, np.array([0, 1, 0])),
                   MonoLinComb(10, np.array([0, 0, 1]))])

    pOut = barycentric_subdivide(pIn, return_offset_separately=False, preserve_scale=False, debug=True)

    print(f"pOut\n{pOut}")

    pOut_expected = LinComb([MonoLinComb(11, [0, 0, 1]),
                             MonoLinComb(6, [1, 0, 1]),
                             MonoLinComb(-7, [1, 1, 1])])

    assert pOut == pOut_expected
    assert str(pOut) == str(pOut_expected)

#def test_BarycentricSubdivide_no_split_preserve_scale():
#    print("###########################################")
#
#    subdivide = BarycentricSubdivide("pIn","pOut", "pOut")
#
#    input_dict = {"pIn": [(-1, np.array([1, 0, 0])),
#                        (-7, np.array([0, 1, 0])),
#                        (10, np.array([0, 0, 1]))],
#                  "metadata1": "moo1",
#                  "metadata2": "moo2",
#                  "metadata3": "moo3",
#                  }
#    enc = subdivide.encode(input_dict, debug=True)
#    print(f"enc\n{enc}")
#    expected = "{'pOut': [(11, array([Fraction(0, 1), Fraction(0, 1), Fraction(1, 1)], dtype=object)), (12, array([Fraction(1, 2), Fraction(0, 1), Fraction(1, 2)], dtype=object)), (-21, array([Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)], dtype=object))]}"
#    assert str(enc) == expected
#
#
#def test_BarycentricSubdivide_split_preserve_scale():
#    print("###########################################")
#
#    subdivide = BarycentricSubdivide("pIn","pOut", "qOut")
#
#    input_dict = {"pIn": [(-1, np.array([1, 0, 0])),
#                        (-7, np.array([0, 1, 0])),
#                        (10, np.array([0, 0, 1]))],
#                  "metadata1": "moo1",
#                  "metadata2": "moo2",
#                  "metadata3": "moo3",
#                  }
#    enc = subdivide.encode(input_dict, debug=True)
#    print(f"enc\n{enc}")
#    expected = "{'pOut': [(11, array([Fraction(0, 1), Fraction(0, 1), Fraction(1, 1)], dtype=object)), (12, array([Fraction(1, 2), Fraction(0, 1), Fraction(1, 2)], dtype=object))], 'qOut': [(-21, array([Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)], dtype=object))]}"
#    assert str(enc) == expected

def test_MonoLinComb():
    a = MonoLinComb(4, [[1,2],[3,4]])
    b = MonoLinComb(4, np.array([[1,2],[3,4]]))
    c = MonoLinComb(4, np.array([[1,2],[3,5]]))
    d = MonoLinComb(4, np.array([[1,2],[3,4],[5,6]]))

    assert a == b
    assert a != c
    assert a != d
    assert b != c
    assert b != d
    assert c != d

def test_LinComb():
    a = LinComb(MonoLinComb(4, [[1,2],[3,4]]))
    b = LinComb(MonoLinComb(4, np.array([[1,2],[3,4]])))
    c = LinComb(MonoLinComb(4, np.array([[1,2],[3,5]])))
    d = LinComb(MonoLinComb(4, np.array([[1,2],[3,4],[5,6]])))

    assert a == b
    assert a != c
    assert a != d
    assert b != c
    assert b != d
    assert c != d

    l1 = LinComb([MonoLinComb(11, np.array([0, 0, 1])),
                  MonoLinComb(6, np.array([1, 0, 1]))])
    l2 = LinComb([MonoLinComb(11, np.array([0, 0, 1])),
                  MonoLinComb(6, np.array([1, 0, 1]))])

    assert l1 == l2


def test_array_to_lin_comb():
    print("###########################################")

    arr = np.asarray([[ 4, 2],
                      [-3, 5],
                      [ 8, 9],
                      [ 2 ,7]])

    enc = array_to_lin_comb(arr)

    expected = LinComb([MonoLinComb(4, [[1,0]
  ,  [0,0]
  ,  [0,0]
  ,  [0,0]]), MonoLinComb(2, [[0,1]
  ,  [0,0]
  ,  [0,0]
  ,  [0,0]]), MonoLinComb(-3, [[0,0]
  ,  [1,0]
  ,  [0,0]
  ,  [0,0]]), MonoLinComb(5, [[0,0]
  ,  [0,1]
  ,  [0,0]
  ,  [0,0]]), MonoLinComb(8, [[0,0]
  ,  [0,0]
  ,  [1,0]
  ,  [0,0]]), MonoLinComb(9, [[0,0]
  ,  [0,0]
  ,  [0,1]
  ,  [0,0]]), MonoLinComb(2, [[0,0]
  ,  [0,0]
  ,  [0,0]
  ,  [1,0]]), MonoLinComb(7, [[0,0]
  ,  [0,0]
  ,  [0,0]
  ,  [0,1]])])
    not_expected = LinComb([MonoLinComb(4, [[1,0]
  ,  [0,0]
  ,  [0,0]
  ,  [0,0]]), MonoLinComb(2, [[0,1]
  ,  [0,0]
  ,  [0,0]
  ,  [0,0]]), MonoLinComb(-3, [[0,11110]
  ,  [1,0]
  ,  [0,0]
  ,  [0,0]]), MonoLinComb(5, [[0,0]
  ,  [0,1]
  ,  [0,0]
  ,  [0,0]]), MonoLinComb(8, [[0,0]
  ,  [0,0]
  ,  [1,0]
  ,  [0,0]]), MonoLinComb(9, [[0,0]
  ,  [0,0]
  ,  [0,1]
  ,  [0,0]]), MonoLinComb(2, [[0,0]
  ,  [0,0]
  ,  [0,0]
  ,  [1,0]]), MonoLinComb(7, [[0,0]
  ,  [0,0]
  ,  [0,0]
  ,  [0,1]])])
    print(f"=======================\nArray to lin comb made encoded")
    print(arr)
    print("to")
    print(f"{enc}")
    assert enc == expected
    assert str(enc) == str(expected)
    assert enc != not_expected
    assert str(enc) != str(not_expected)


#def test_Chain_1():
#    print("###########################################")
#    simplex1_bit = Chain([
#        BarycentricSubdivide("set", "first_diffs", "offset"),
#        BarycentricSubdivide("first_diffs", "second_diffs", "second_diffs", pass_forward="offset")
#    ])
#
#    input_dict = {
#                  "set": [(-1, np.array([1, 0, 0])),
#                          (-7, np.array([0, 1, 0])),
#                          (10, np.array([0, 0, 1]))],
#                  "metadata1": "moo1",
#                  "metadata2": "moo2",
#                  "metadata3": "moo3",
#                  }
#    enc = simplex1_bit.encode(input_dict, debug=True)
#    print(f"=======================\nSimplex1 as a chain encoded")
#    print(input_dict)
#    print("to")
#    print(f"{enc}")
#
#
#def test_simplex_1_initial_encoding_phase():
#    print("###########################################")
#    simplex1_different_bit = Chain([
#        array_to_lin_comb(input_array_name="set", output_lin_comb_name="lin_comb_0"),
#        BarycentricSubdivide("lin_comb_0", "lin_comb_1_first_diffs", "offset", preserve_scale=False),
#        BarycentricSubdivide("lin_comb_1_first_diffs", "lin_comb_2_second_diffs",
#                             "lin_comb_2_second_diffs", pass_forward="offset", pass_backward="offset", preserve_scale=False),
#        MergeLinCombs(["lin_comb_2_second_diffs", "offset"], "lin_comb_3"),
#    ])
#
#    input_dict = {
#                  "set" : np.asarray([[ 4, 2],
#                                      [-3, 5],
#                                      [ 8, 9],
#                                      [ 2 ,7]]),
#                  "metadata1": "moo1",
#                  "metadata2": "moo2",
#                  "metadata3": "moo3",
#                  }
#    enc = simplex1_different_bit.encode(input_dict, debug=True)
#    print(f"=======================\nSimplex1 as a chain encoded")
#    print(numpy_array_of_frac_to_str(input_dict["set"]))
#    print("to")
#    #print(f"{enc}")
#
#    lin_comb_3 = enc["lin_comb_3"]
#    #print(f"Note that lin_comb_3 is")
#    pretty_print_lin_comb(lin_comb_3)
#    print("and the (non-offset) differences are")
#    [ print(numpy_array_of_frac_to_str(tmp:=b-a), " with ", np.sum(tmp)," ones in it") for a,b in list(pairwise( [a for _,a in lin_comb_3 ]))[:-1] ]
#
#
#    print("###########################################")


