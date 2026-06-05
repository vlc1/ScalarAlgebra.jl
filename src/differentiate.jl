function differentiate end

differentiate(a::AbstractScalar, b::ScalarSym) =
    ScalarZero(_jacobian_type(eltype(a), eltype(b)))

differentiate(a::AbstractScalar, b::ScalarRef) =
    ScalarRef(differentiate(a, b.arr), a.indices)

# ScalarSym: identical symbols and `eltype`s
differentiate(::T, ::T) where {T <: ScalarSym} = ScalarOne(eltype(T))

# ScalarCall
_differentiate_args(::Tuple{}, _) = ()
_differentiate_args((a, as...)::Tuple{AbstractScalar, Vararg}, b) =
    differentiate(a, b), _differentiate_args(as, b)...

differentiate(a::ScalarCall, b::ScalarSym) =
    _differentiate_call(a.fn, a.args, _differentiate_args(a.args, b))

# addition
_differentiate_call(::typeof(+), _, ders) = ScalarCall(+, ders)

# subtraction
_differentiate_call(::typeof(-), _, ders) = ScalarCall(-, ders)

# multiplication
_differentiate_call(::typeof(*), args, ders) = _differentiate_mul(args, ders)

#_differentiate_mul((u, _)::Tuple{ScalarConst, AbstractScalar}, (_, dv)) = u * dv
#_differentiate_mul((_, v)::Tuple{AbstractScalar, ScalarConst}, (du, _)) = du * v
_differentiate_mul(((u, v), (du, dv))::Vararg{NTuple{2, AbstractScalar}, 2}) =
    du * v + u * dv

# right-division
_differentiate_call(::typeof(/), args, ders) = _differentiate_rdiv(args, ders)

_differentiate_rdiv(((u, v), (du, dv))::Vararg{NTuple{2, AbstractScalar}, 2}) =
    (du - (u / v) * dv) / v

# left-division
_differentiate_call(::typeof(\), args, ders) = _differentiate_ldiv(args, ders)

_differentiate_ldiv(((u, v), (du, dv))::Vararg{NTuple{2, AbstractScalar}, 2}) =
    u \ (dv - du * (u \ v))

# ScalarRef
differentiate(a::ScalarRef, b::ScalarSym) =
    ScalarRef(differentiate(a.arr, b), (a.indices..., _colon_pad(eltype(b))...))
