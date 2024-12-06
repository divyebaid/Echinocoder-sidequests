# Echinocoder

This is a library contains functions which are able to perform:

  * Embeddings of real [symmetric product spaces](https://en.wikipedia.org/wiki/Symmetric_product_(topology)), $SP^n(\mathbb R^k)$.  These are continuous bijective mappings of multisets of size $n$ containing vectors in $\mathbb{R}^k$ into $\mathbb R^k$ for some $k$.

  * Embeddings of $\mathbb R^k$ in which $\vec x\in\mathbb R^k$ is identified with $-\vec x$.  I am not sure what these are really supposed to be called. This libarary currently calls them [real projective spaces](https://en.wikipedia.org/wiki/Real_projective_space) but that might be an abuse of terminology.

Most embedders work only reals inputs and generate only real embeddings as that's the whole purpose of the libarary. However, some embedders will accept complex numbers as inputs and can generate complex numbers as outputs.  Where this is the case it is not always documented. Some of the embedders which can process complex inputs and outputs are nonetheless used (in complex mode) as steps in the implementation of other embedders.  The capacity for some embedders to process complex numbers such routines should be considered private (unexposed) even if technically visible. This is to allow interface standardisation.

## $SP^n(\mathbb R^k)$ -- i.e. multiset embedders:

All these are (or should be) instances of [MultisetEmbedder](MultisetEmbedder.py).


* The [Simplicial Complex](https://en.wikipedia.org/wiki/Simplicial_complex) embedder works for any $n$ and $k$ and embeds into $2 n k+1$ reals. ([embedder source](C0HomDeg1_simplicialComplex_embedder_1_for_array_of_reals_as_multiset.py)) 
* [This algorithm](C0HomDeg1_conjectured_dotting_embedder_for_array_of_reals_as_multiset.py) based on the [dotting encoder](C0HomDeg1_dotting_encoder_for_array_of_reals_as_multiset.py) is CONJECTURED (but not proved) to be an embedder. It has $ORDER(n,k) = O(n k \log n)$. ([embedder source](C0HomDeg1_conjectured_dotting_embedder_for_array_of_reals_as_multiset.py)). 
* The polynomial embedders 
have order $O(n m^2)$ in general, but happen to be efficient (i.e. embed into $nk$ reals) for $k=1$ or $k=2$ but 
. (embedder sources ([for multisets of vectors](Cinf_numpy_polynomial_embedder_for_array_of_reals_as_multiset.py)) and ([for multisets of reals](Cinf_numpy_polynomial_embedder_for_list_of_reals_as_multiset.py)))
* The (vanilla) busar embedder has order $O(n^2 k)$.  Indeed, the exact order is  $ORDER(n,k) = n + (k-1) n (n+1)/2$. ([embedder source](Cinf_sympy_bursar_embedder_for_array_of_reals_as_multiset.py))
* The 'even' busar embedder has order $Binom(n+k,n)-1$. While this embedder is very inefficient, it does not treat any components in the $k$-space differently than any other.  ([embedder source](Cinf_sympy_bursar_embedder_for_array_of_reals_as_multiset.py))
* If one were to use the busar embedder when $k\ge n$ and the polynomial embedder when $n\ge k$ then one would have, in effect, a single method of order $O((nkk^{\frac 3 2})$. [Check this statement! It is probably not true!]

## Obsolete/Retired/Historical embedders:
* An early (nonlinear) [Simplicial Complex](https://en.wikipedia.org/wiki/Simplicial_complex) embedder 
([embedder source](Historical/C0_simplicialComplex_embedder_1_for_array_of_reals_as_multiset.py))
workedfor any $n$ and $k$ and embedded into $4 n k+1$ reals. 
In principle the method could embed into just $2 n k + 1$ reals.  However, an implementation choice which was expected to make the outputs more stable leads to the number of outputs being $4 n k + 1$ instead.

## Embedders which work on $SP^m(\mathbb R)$ only (i.e. $k=1$ is hard coded and inputs are 1-d lists  not 2-d arrays)

* The sorting embedder is efficient (i.e. embeds into $n$ reals for any $n$. ([embedder source](C0_sorting_embedder_for_list_of_reals_as_multiset.py))

## What this library is calling $RP(\mathbb R^m)$ ([real projective space](https://en.wikipedia.org/wiki/Real_projective_space)) embedders:

* By setting $n=2$ and embedding the multiset $\left\\{\vec x,-\vec x\right\\}$ with $\vec x$ in $R^m$ one can use the bursar embedder to embed something this library calls $RP^m$ (which is possibly an abuse of the notation for real projective space of order $m$).  This $RP^m$ embedding would (for the (vanilla) bursar embedder) naively therefore be of size $2+(m-1)2(2+1)/2 = 2+3(m-1)$.  However, since all $m$ terms of order 1 in the auxiliary variable $y$ always disappear for multisets of this sort, the coefficients of those terms do not need to be recorded. This leaves only $2m-1$ reals needing to be recorded in the embedding for $RP^m$.  A method named [Cinf_numpy_regular_embedder_for_list_of_realsOrComplex_as_realOrComplexprojectivespace](Cinf_numpy_regular_embedder_for_list_of_realsOrComplex_as_realOrComplexprojectivespace.py) implements this method. It is order $2n-1$ when $n>0$.
* A small optimisation of the above method (implemented as [Cinf_numpy_complexPacked_embedder_for_list_of_reals_as_realprojectivespace](Cinf_numpy_complexPacked_embedder_for_list_of_reals_as_realprojectivespace.py))  reduces the by one when $n>0$ and $n$ is even.


## Testing/examples

[example.py](example.py) is a simple example showing how one of the embedders could be used.

[test.py](test.py) excercises some of the embedders. If they all work the script should end with a message saying something like 

"----------------------------------------------------------------------
Ran 1 test in 0.165s

OK
"

[test_PKH_alg.py](test_PKH_alg.py) contains another set of unit tests, mainly intended to exercise the Simplicial Complex embedder.

## References:

Neither the [Don Davis papers](https://www.lehigh.edu/~dmd1/toppapers.html) nor [Don Davis immersion list](https://www.lehigh.edu/~dmd1/imms.html) has been used to create this library. Both may, however, be useful references and sources of other references, so some are cached in the [DOCS](DOCS) directory.
