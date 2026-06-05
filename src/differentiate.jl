"""
    differentiate(a::AbstractScalar, b::ScalarSym) -> AbstractScalar

Symbolic differentiation of scalar expression `a` with respect to symbol `b`.
Returns the Jacobian as an `AbstractScalar` whose `eltype` is determined by
`_jacobian_type(eltype(a), eltype(b))`:

- `(Number, Number)` → `promote_type(S, T)` (scalar derivative)
- `(SVector{M}, SVector{N})` → `SMatrix{M,N}` (full Jacobian matrix)
- `(Number, SVector{N})` → `SMatrix{1,N}` (row / gradient)
- `(SVector{M}, Number)` → `SVector{M}` (column / directional)

## Outer-product ordering in chain rules

The gradient of a scalar w.r.t. `SVector{N}` is a row (`SMatrix{1,N}`). When this row
Jacobian `du` is combined with an array value `v::SVector{N}`, operand order is critical:

- `du * v` = `SMatrix{1,N} × SVector{N}` → inner product → `SVector{1}` (wrong shape)
- `v * du` = `SVector{N} × SMatrix{1,N}` → outer product → `SMatrix{N,N}` (correct)

Two rules apply this swap via more-specific overloads:

| operator | case | formula |
|----------|------|---------|
| `*` | `scalar × array` | `v * du + u * dv` |
| `\\` | `scalar \\ array` | `u \\ (dv - (u\\v) * du)` |

`/` is unaffected: `(u/v)` is already the array and sits on the left of `*` in its formula.
"""
function differentiate end

differentiate(a::AbstractScalar, b::ScalarSym) =
    ScalarZero(_jacobian_type(eltype(a), eltype(b)))

differentiate(a::AbstractScalar, b::ScalarRef) =
    ScalarRef(differentiate(a, b.arr), (_colon_pad(eltype(a))..., b.indices...))

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

_differentiate_mul(
    (u, v)::Tuple{AbstractScalar{<:Number}, AbstractScalar{<:AbstractArray}},
    (du, dv)::NTuple{2, AbstractScalar}) = v * du + u * dv

_differentiate_mul(((u, v), (du, dv))::Vararg{NTuple{2, AbstractScalar}, 2}) =
    du * v + u * dv

# right-division
_differentiate_call(::typeof(/), args, ders) = _differentiate_rdiv(args, ders)

_differentiate_rdiv(((u, v), (du, dv))::Vararg{NTuple{2, AbstractScalar}, 2}) =
    (du - (u / v) * dv) / v

# left-division
_differentiate_call(::typeof(\), args, ders) = _differentiate_ldiv(args, ders)

_differentiate_ldiv(
    (u, v)::Tuple{AbstractScalar{<:Number}, AbstractScalar{<:AbstractArray}},
    (du, dv)::NTuple{2, AbstractScalar}) = u \ (dv - (u \ v) * du)

_differentiate_ldiv(((u, v), (du, dv))::Vararg{NTuple{2, AbstractScalar}, 2}) =
    u \ (dv - du * (u \ v))

# ScalarRef
_wrap_scalar_index(idx::ScalarConst{<:Integer}) = ScalarConst(SVector(idx.val))
_wrap_scalar_index(idx) = idx

# When ref has scalar output and sym has array input, integer indices collapse the row
# dimension of the Jacobian (M[i,:] → SVector vs M[SA[i],:] → SMatrix{1,N}).
# Wrap them to preserve shape.
_ref_indices(indices, ::Type{<:Number}, ::Type{<:AbstractArray}) =
    map(_wrap_scalar_index, indices)
_ref_indices(indices, _, _) = indices

differentiate(a::ScalarRef, b::ScalarSym) =
    ScalarRef(differentiate(a.arr, b),
        (_ref_indices(a.indices, eltype(a), eltype(b))..., _colon_pad(eltype(b))...))
