"""
    simplify(s::AbstractScalar) -> AbstractScalar

Structurally simplify a scalar expression tree (post-order, single pass).
"""
function simplify end

simplify(sc::AbstractScalar) = sc

# ScalarCall
_simplify_args(::Tuple{}) = ()
_simplify_args((x, xs...)::Tuple{AbstractScalar, Vararg}) = simplify(x), _simplify_args(xs)...

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

# subtractive identities
_simplify_call(::typeof(-), args) = _simplify_sub(args)

_simplify_sub((a,)::Tuple{AbstractScalar}) = -a
_simplify_sub((a, b)::Tuple{AbstractScalar, AbstractScalar}) = a - b

_simplify_sub((a, _)::Tuple{AbstractScalar, ScalarZero}) = a
_simplify_sub((_, b)::Tuple{ScalarZero, AbstractScalar}) = -b
_simplify_sub((a, b)::Tuple{ScalarZero, ScalarZero}) =
    ScalarZero(Base.promote_op(+, eltype(a), eltype(b)))

_simplify_sub((a, b)::Tuple{ScalarConst, ScalarConst}) =
    ScalarConst(a.val - b.val)
_simplify_sub((a, b)::Tuple{ScalarOne, ScalarConst}) =
    ScalarConst(one(eltype(a)) - b.val)
_simplify_sub((a, b)::Tuple{ScalarConst, ScalarOne}) =
    ScalarConst(a.val - one(eltype(b)))

# double negation
_simplify_sub((a,)::Tuple{ScalarCall{typeof(-), <: Tuple{AbstractScalar}}}) = only(a.args)

# multiplicative identities
_simplify_call(::typeof(*), args) = _simplify_mul(args)

_simplify_mul((a, b)::Tuple{AbstractScalar, AbstractScalar}) = a * b

_simplify_mul((a, b)::Tuple{AbstractScalar, ScalarZero}) =
    ScalarZero(Base.promote_op(*, eltype(a), eltype(b)))
_simplify_mul((a, b)::Tuple{ScalarZero, AbstractScalar}) =
    ScalarZero(Base.promote_op(*, eltype(a), eltype(b)))
_simplify_mul((a, b)::Tuple{ScalarZero, ScalarZero}) =
    ScalarZero(Base.promote_op(*, eltype(a), eltype(b)))

_simplify_mul((a, b)::Tuple{AbstractScalar, ScalarOne}) =
    Base.promote_op(*, eltype(a), eltype(b)) === eltype(a) ? a : a * b
_simplify_mul((a, b)::Tuple{ScalarOne, AbstractScalar}) =
    Base.promote_op(*, eltype(a), eltype(b)) === eltype(b) ? b : a * b
_simplify_mul((a, b)::Tuple{ScalarOne, ScalarOne}) =
    ScalarOne{Base.promote_op(*, eltype(a), eltype(b))}()

_simplify_mul((a, b)::Tuple{ScalarConst, ScalarConst}) =
    ScalarConst(a.val * b.val)
_simplify_mul((a, b)::Tuple{ScalarZero, ScalarOne}) =
    ScalarZero(Base.promote_op(*, eltype(a), eltype(b)))
_simplify_mul((a, b)::Tuple{ScalarOne, ScalarZero}) =
    ScalarZero(Base.promote_op(*, eltype(a), eltype(b)))

# right-division-based identities
_simplify_call(::typeof(/), args) = _simplify_rdiv(args)

_simplify_rdiv((a, b)::Tuple{AbstractScalar, AbstractScalar}) = a / b

_simplify_rdiv((a, b)::Tuple{AbstractScalar, ScalarOne}) =
    Base.promote_op(/, eltype(a), eltype(b)) === eltype(a) ? a : a / b
_simplify_rdiv((a, b)::Tuple{ScalarZero, AbstractScalar}) =
    ScalarZero(Base.promote_op(/, eltype(a), eltype(b)))

_simplify_rdiv((a, b)::Tuple{ScalarZero, ScalarOne}) =
    ScalarZero(Base.promote_op(/, eltype(a), eltype(b)))

_simplify_rdiv(::NTuple{2, T}) where {T <: ScalarSym} =
    ScalarOne{Base.promote_op(/, eltype(T), eltype(T))}()
_simplify_rdiv((a, b)::Tuple{ScalarConst, ScalarConst}) =
    ScalarConst(a.val / b.val)
_simplify_rdiv((a, b)::Tuple{ScalarOne, ScalarOne}) =
    ScalarOne{Base.promote_op(/, eltype(a), eltype(b))}()

# left-division-based identities
_simplify_call(::typeof(\), args) = _simplify_ldiv(args)

_simplify_ldiv((a, b)::Tuple{AbstractScalar, AbstractScalar}) = a \ b

_simplify_ldiv((a, b)::Tuple{ScalarOne, AbstractScalar}) =
    Base.promote_op(\, eltype(a), eltype(b)) === eltype(b) ? b : a \ b
_simplify_ldiv((a, b)::Tuple{AbstractScalar, ScalarZero}) =
    ScalarZero(Base.promote_op(\, eltype(a), eltype(b)))

_simplify_ldiv((a, b)::Tuple{ScalarOne, ScalarZero}) =
    ScalarZero(Base.promote_op(\, eltype(a), eltype(b)))

_simplify_ldiv(::NTuple{2, T}) where {T <: ScalarSym} =
    ScalarOne{Base.promote_op(\, eltype(T), eltype(T))}()
_simplify_ldiv((a, b)::Tuple{ScalarConst, ScalarConst}) =
    ScalarConst(a.val \ b.val)
_simplify_ldiv((a, b)::Tuple{ScalarOne, ScalarOne}) =
ScalarOne{Base.promote_op(\, eltype(a), eltype(b))}()

# ScalarRef
simplify(sc::ScalarRef) = _simplify_ref(simplify(sc.arr), _simplify_args(sc.indices))

_simplify_ref(arr::AbstractScalar{<: Number}, _) = arr
_simplify_ref(arr::AbstractScalar, indices) = ScalarRef(arr, indices)

_simplify_ref(a::ScalarConst, indices::Tuple{Vararg{ScalarConst}}) =
    ScalarConst(a.val[getfield.(indices, :val)...])
_simplify_ref(::ScalarZero{T}, indices::Tuple{Vararg{ScalarConst}}) where {T} =
    (checkbounds(zero(T), getfield.(indices, :val)...); ScalarZero{eltype(T)}())
_simplify_ref(::ScalarOne{T}, indices::Tuple{Vararg{ScalarConst}}) where {T} =
    ScalarConst(one(T)[getfield.(indices, :val)...])

_simplify_ref(call::ScalarCall, indices) =
    _simplify_ref_call(call.fn, call.args, indices)

_simplify_ref_call(::typeof(+), args, indices) =
    _simplify_call(+, map(x -> _simplify_ref(x, indices), args))
_simplify_ref_call(::typeof(-), args, indices) =
    _simplify_call(-, map(x -> _simplify_ref(x, indices), args))
_simplify_ref_call(::typeof(*), args, indices) =
    _simplify_ref_mul(args, indices)
_simplify_ref_call(::typeof(/), args, indices) =
    _simplify_ref_rdiv(args, indices)
_simplify_ref_call(::typeof(\), args, indices) =
    _simplify_ref_ldiv(args, indices)
_simplify_ref_call(fn, args, indices) = ScalarRef(ScalarCall(fn, args), indices)

_simplify_ref_mul((a, b)::Tuple{AbstractScalar{<: Number}, AbstractScalar}, indices) =
    _simplify_call(*, (a, _simplify_ref(b, indices)))
_simplify_ref_mul((a, b)::Tuple{AbstractScalar, AbstractScalar{<: Number}}, indices) =
    _simplify_call(*, (_simplify_ref(a, indices), b))
_simplify_ref_mul((a, b)::Tuple{AbstractScalar, AbstractScalar}, indices) =
    ScalarRef(ScalarCall(*, (a, b)), indices)

_simplify_ref_rdiv((a, b)::Tuple{AbstractScalar, AbstractScalar{<: Number}}, indices) =
    _simplify_call(/, (_simplify_ref(a, indices), b))
_simplify_ref_rdiv((a, b)::Tuple{AbstractScalar, AbstractScalar}, indices) =
    ScalarRef(ScalarCall(/, (a, b)), indices)

_simplify_ref_ldiv((a, b)::Tuple{AbstractScalar{<: Number}, AbstractScalar}, indices) =
    _simplify_call(\, (a, _simplify_ref(b, indices)))
_simplify_ref_ldiv((a, b)::Tuple{AbstractScalar, AbstractScalar}, indices) =
    ScalarRef(ScalarCall(\, (a, b)), indices)
