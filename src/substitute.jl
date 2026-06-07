# `substitute` is owned by AlgebraCore; these methods extend it for AbstractScalar.

"""
    substitute(s::AbstractScalar, pairs::NamedTuple) -> AbstractScalar

Replace each `ScalarSym` named in `pairs` by `asscalar(pairs[name])`, returning a
scalar expression (unlike [`materialize`](@ref), which evaluates to a value).

A binding may be a raw value (wrapped as a `ScalarConst`) or another
`AbstractScalar` (spliced in as-is, via `asscalar`). Substitution is **partial**:
symbols absent from `pairs` are kept. The tree is rebuilt structurally with no
simplification — compose with `simplify`/`materialize` afterward.
"""
substitute(sc::AbstractScalar, ::NamedTuple) = sc

substitute(sc::ScalarSym{S}, pairs::NamedTuple) where {S} =
    haskey(pairs, S) ? asscalar(pairs[S]) : sc

# type-stable tuple recursion (mirrors _simplify_args / _differentiate_args)
_substitute_args(::Tuple{}, ::NamedTuple) = ()
_substitute_args((a, as...)::Tuple, pairs::NamedTuple) =
    (substitute(a, pairs), _substitute_args(as, pairs)...)

substitute(sc::ScalarCall, pairs::NamedTuple) =
    ScalarCall(sc.fn, _substitute_args(sc.args, pairs))
substitute(sc::ScalarRef, pairs::NamedTuple) =
    ScalarRef(substitute(sc.arr, pairs), _substitute_args(sc.indices, pairs))
