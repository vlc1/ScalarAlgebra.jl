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

# additive identities
_simplify_call(::typeof(+), args) = _simplify_add(args)

_simplify_add((a,)::Tuple{AbstractScalar}) = a
_simplify_add((a, b)::Tuple{AbstractScalar, AbstractScalar}) = a + b

_simplify_add((a, _)::Tuple{AbstractScalar, ScalarZero}) = a
_simplify_add((_, b)::Tuple{ScalarZero, AbstractScalar}) = b
_simplify_add((a, b)::Tuple{ScalarZero, ScalarZero}) =
    ScalarZero(Base.promote_op(+, eltype(a), eltype(b)))

_simplify_add((a, b)::Tuple{ScalarConst, ScalarConst}) =
    ScalarConst(a.val + b.val)
_simplify_add((a, b)::Tuple{ScalarOne, ScalarConst}) =
    ScalarConst(one(eltype(a)) + b.val)
_simplify_add((a, b)::Tuple{ScalarConst, ScalarOne}) =
    ScalarConst(a.val + one(eltype(b)))

# substractive identities
_simplify_call(::typeof(-), args) = _simplify_sub(args)

_simplify_sub((a,)::Tuple{AbstractScalar}) = -a
_simplify_sub((a, b)::Tuple{AbstractScalar, AbstractScalar}) = a - b

# double negation: -(-a) = a
_simplify_sub((a,)::Tuple{ScalarCall{typeof(-), <: Tuple{AbstractScalar}}}) = only(a.args)

_simplify_sub((a, _)::Tuple{AbstractScalar, ScalarZero}) = a
_simplify_sub((_, b)::Tuple{ScalarZero, AbstractScalar}) =
    _simplify_sub((b,))
_simplify_sub((a, b)::Tuple{ScalarZero, ScalarZero}) =
    ScalarZero(Base.promote_op(+, eltype(a), eltype(b)))

_simplify_sub((a, b)::Tuple{ScalarConst, ScalarConst}) =
    ScalarConst(a.val - b.val)
_simplify_sub((a, b)::Tuple{ScalarOne, ScalarConst}) =
    ScalarConst(one(eltype(a)) - b.val)
_simplify_sub((a, b)::Tuple{ScalarConst, ScalarOne}) =
    ScalarConst(a.val - one(eltype(b)))

# ScalarRef
