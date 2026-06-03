"""
    simplify(s::AbstractScalar) -> AbstractScalar

Structurally simplify a scalar expression tree (post-order, single pass).
"""
function simplify end

simplify(sc::AbstractScalar) = sc

# ScalarCall
_simplify_args(::Tuple{}) = ()
_simplify_args((x, xs...)::Tuple{Any, Vararg}) = simplify(x), _simplify_args(xs)...

simplify(sc::ScalarCall) = _simplify_call(sc.fn, _simplify_args(sc.args))

## additive identities
_simplify_call(::typeof(+), args) = _simplify_add(args)
_simplify_call(::typeof(+), (arg,)::Tuple{AbstractScalar}) = arg

_simplify_add((a, _)::Tuple{AbstractScalar, ScalarZero}) = a
_simplify_add((_, b)::Tuple{ScalarZero, AbstractScalar}) = b
_simplify_add((a, b)::Tuple{ScalarZero, ScalarZero}) =
    ScalarZero(Base.promote_op(+, eltype(a), eltype(b)))

# ScalarRef
