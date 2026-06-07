"""
    differentiate(f::AbstractScalar, x::ScalarSym) -> AbstractScalar
    differentiate(f::AbstractScalar, b::ScalarRef) -> AbstractScalar

Dense Jacobian of `f`, reconstructed from [`pushforward`](@ref) by seeding the
input tangent space:

- scalar input (`eltype(x) <: Number`): a single JVP seeded with `1` (a scalar
  self-derivative folds to `ScalarOne`). Output shape follows `eltype(f)`
  (scalar → scalar, `SVector{M}` → column).
- vector input (`eltype(x) == SVector{N}`): one JVP per basis direction `eⱼ`,
  the columns assembled into the dense Jacobian — `SMatrix{M,N}` for `SVector{M}`
  output, `SMatrix{1,N}` (a row) for scalar output.
- `differentiate(f, b::ScalarRef)` is `∂f/∂(b.arr[b.indices])`: one JVP seeded
  with the unit tangent selecting that element (the `b.indices`-th column of the
  identity).

(`differentiate` itself is owned by AlgebraCore; these methods extend it.)
"""
differentiate(f::AbstractScalar, x::ScalarSym{S, T}) where {S, T <: Number} =
    pushforward(f, x, ScalarOne(T))

# vector self-derivative: keep the clean structural identity rather than
# reconstructing it column by column (strictly more specific than the generic
# vector method below, so no dispatch ambiguity).
differentiate(::ScalarSym{S, SVector{N, T}}, ::ScalarSym{S, SVector{N, T}}) where {S, N, T} =
    ScalarOne(SVector{N, T})

# vector input SVector{N}: seed each basis direction with a structural one-hot,
# assemble the columns. Type-level (StaticInt) seed/row indices let the one-hot
# folds (`src/simplify.jl`) collapse the Jacobian to its sparse structural form.
differentiate(f::AbstractScalar, x::ScalarSym{S, SVector{N, T}}) where {S, N, T} =
    _assemble_jacobian(_jvp_columns(f, x, SVector{N, T}), eltype(f), Val(N))

# JVP columns, one per basis direction. @generated so each seed's hot index is a
# literal type parameter (the columns have distinct OneHotScalar{N,K} types).
@generated function _jvp_columns(f, x, ::Type{SVector{N, T}}) where {N, T}
    cols = (:(pushforward(f, x, OneHotScalar{$N, $K}())) for K in 1:N)
    :(($(cols...),))
end

# derivative w.r.t. one array element: a single JVP seeded with the unit there
differentiate(f::AbstractScalar, b::ScalarRef) =
    pushforward(f, b.arr, _unit_seed(b.arr, b.indices))

# unit tangent selecting element `k` of an SVector-valued symbol: I[:, k] = e_k.
# A static index seeds with a structural one-hot so the derivative folds to
# ScalarOne/ScalarZero (e.g. d(v[static i])/d(v[static j]) → δ_ij); a runtime or
# symbolic index keeps the I[:, k] slice (folds to a Bool ScalarConst).
_unit_seed(::AbstractScalar{SVector{N, T}}, ::Tuple{ScalarConst{StaticInt{K}}}) where {N, T, K} =
    OneHotScalar{N, K}()
_unit_seed(arr::AbstractScalar{<:SVector}, (k,)::Tuple{AbstractScalar}) =
    ScalarRef(ScalarOne(eltype(arr)), (ScalarConst(Colon()), k))

# Assemble the JVP columns into the dense Jacobian. This is the single place
# where output shape is materialized.
# scalar output → 1×N row
_assemble_jacobian(cols::NTuple{N, AbstractScalar}, ::Type{<:Number}, ::Val{N}) where {N} =
    SMatrix{1, N}(cols...)
# SVector{M} output → M×N matrix, column-major (entry l ↦ cols[col][row]).
# @generated so column/row indices are literals — rows index with StaticInt so
# one-hot columns fold to ScalarOne/ScalarZero, type-stably regardless of how
# complex the column expressions are.
@generated function _assemble_jacobian(cols::NTuple{N, AbstractScalar}, ::Type{<:SVector{M}}, ::Val{N}) where {M, N}
    entries = (:(cols[$(cld(l, M))][static($(mod1(l, M)))]) for l in 1:M*N)
    :(SMatrix{$M, $N}($(entries...)))
end
