"""
    pushforward(f::AbstractScalar, x::ScalarSym, ẋ::AbstractScalar) -> AbstractScalar

Forward-mode directional derivative (Jacobian–vector product) of `f` with respect
to symbol `x`, along the input tangent `ẋ`.

The result lives in `f`'s own value space (`eltype(pushforward(f, …)) ==
eltype(f)`): the tangent of a scalar-valued node is a scalar, the tangent of a
vector-valued node is a vector. Consequently **every product in the chain and
Leibniz rules below is an ordinary value-space operation** — there is no covector,
no `SMatrix{1,N}` row, and no operand-order bookkeeping. Correctness is structural.

The dense Jacobian ([`differentiate`](@ref)) is reconstructed from this primitive
by seeding the input tangent space with basis directions.
"""
function pushforward end

# leaves --------------------------------------------------------------------
# Same symbol: the seed itself. Different symbol / literal: zero tangent in the
# node's own value space.
pushforward(::T, ::T, ẋ::AbstractScalar) where {T <: ScalarSym} = ẋ
pushforward(f::ScalarSym, ::ScalarSym, ::AbstractScalar) = ScalarZero(eltype(f))
pushforward(f::ScalarConst, ::ScalarSym, ::AbstractScalar) = ScalarZero(eltype(f))
pushforward(f::ScalarZero, ::ScalarSym, ::AbstractScalar) = ScalarZero(eltype(f))
pushforward(f::ScalarOne, ::ScalarSym, ::AbstractScalar) = ScalarZero(eltype(f))
pushforward(f::OneHotScalar, ::ScalarSym, ::AbstractScalar) = ScalarZero(eltype(f))

# interior ------------------------------------------------------------------
pushforward(f::ScalarCall, x::ScalarSym, ẋ::AbstractScalar) =
    _pushforward_call(f.fn, f.args, x, ẋ)

_pf(::Tuple{}, ::ScalarSym, ::AbstractScalar) = ()
_pf((a, as...)::Tuple, x::ScalarSym, ẋ::AbstractScalar) =
    (pushforward(a, x, ẋ), _pf(as, x, ẋ)...)

# linearity
_pushforward_call(::typeof(+), args, x, ẋ) = ScalarCall(+, _pf(args, x, ẋ))
_pushforward_call(::typeof(-), args, x, ẋ) = ScalarCall(-, _pf(args, x, ẋ))

# Leibniz / quotient — all value-space products (no operand-order rules).
_pushforward_call(::typeof(*), (u, v)::NTuple{2, AbstractScalar}, x, ẋ) =
    u * pushforward(v, x, ẋ) + pushforward(u, x, ẋ) * v
_pushforward_call(::typeof(/), (u, v)::NTuple{2, AbstractScalar}, x, ẋ) =
    (pushforward(u, x, ẋ) - (u / v) * pushforward(v, x, ẋ)) / v
_pushforward_call(::typeof(\), (u, v)::NTuple{2, AbstractScalar}, x, ẋ) =
    u \ (pushforward(v, x, ẋ) - (u \ v) * pushforward(u, x, ẋ))

# power with constant exponent:  d(u^c) = c*u^(c-1) * du
_pushforward_call(::typeof(^), (u, c)::Tuple{AbstractScalar, ScalarConst}, x, ẋ) =
    (c * u ^ (c.val - 1)) * pushforward(u, x, ẋ)
_pushforward_call(::typeof(^), (u, _)::Tuple{AbstractScalar, ScalarOne}, x, ẋ) =
    pushforward(u, x, ẋ)
_pushforward_call(::typeof(^), ::Tuple{AbstractScalar, AbstractScalar}, _, _) =
    throw(ArgumentError("pushforward: u^v with non-constant exponent is unsupported"))

# unary nonlinear chain rules:  d f(u) = f'(u) * du
_pushforward_call(::typeof(exp), (u,)::Tuple{AbstractScalar}, x, ẋ) =
    exp(u) * pushforward(u, x, ẋ)
_pushforward_call(::typeof(log), (u,)::Tuple{AbstractScalar}, x, ẋ) =
    pushforward(u, x, ẋ) / u
_pushforward_call(::typeof(sin), (u,)::Tuple{AbstractScalar}, x, ẋ) =
    cos(u) * pushforward(u, x, ẋ)
_pushforward_call(::typeof(cos), (u,)::Tuple{AbstractScalar}, x, ẋ) =
    -sin(u) * pushforward(u, x, ẋ)
_pushforward_call(::typeof(tan), (u,)::Tuple{AbstractScalar}, x, ẋ) =
    pushforward(u, x, ẋ) / cos(u)^2
_pushforward_call(::typeof(sqrt), (u,)::Tuple{AbstractScalar}, x, ẋ) =
    pushforward(u, x, ẋ) / (2 * sqrt(u))
_pushforward_call(::typeof(abs), (u,)::Tuple{AbstractScalar}, x, ẋ) =
    sign(u) * pushforward(u, x, ẋ)
_pushforward_call(::typeof(sign), (u,)::Tuple{AbstractScalar}, _, _) =
    ScalarZero(eltype(u))

# min/max have a value-dependent subgradient — fail loudly.
_pushforward_call(::typeof(min), ::NTuple{2, AbstractScalar}, _, _) =
    throw(ArgumentError("pushforward: min is not differentiable (value-dependent subgradient)"))
_pushforward_call(::typeof(max), ::NTuple{2, AbstractScalar}, _, _) =
    throw(ArgumentError("pushforward: max is not differentiable (value-dependent subgradient)"))

# StaticArray constructor: distribute (constructors are linear in their entries).
_pushforward_call(fn::Type{<:StaticArray}, args, x, ẋ) = fn(_pf(args, x, ẋ)...)

# Indexing is linear in the array; integer indices are independent of `x`.
pushforward(f::ScalarRef, x::ScalarSym, ẋ::AbstractScalar) =
    ScalarRef(pushforward(f.arr, x, ẋ), f.indices)
