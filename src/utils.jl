# Shared concrete-eltype guard, used by every leaf type whose `T` it
# materializes/assembles into.
@inline function _assert_concrete(name::Symbol, ::Type{T}) where {T}
    isconcretetype(T) || throw(ArgumentError(
        "$(name) needs a concrete element type; got $(T). Use e.g. Float64 " *
        "(the default), Float32, or a concrete SVector."))
    return nothing
end

# Map any concrete type to its Bool-shaped counterpart: Number to Bool;
# SArray{S,T,N,L} to SArray{S,Bool,N,L}.
_to_bool_shape(::Type{T}) where {T <: Number} = Bool
_to_bool_shape(::Type{T}) where {T <: StaticArray} = similar_type(T, Bool)

# Inner-constructor guard: T must be concrete *and* Bool-shaped.
@inline function _assert_bool_shape(name::Symbol, ::Type{T}) where {T}
    _assert_concrete(name, T)
    (T === Bool || (T <: AbstractArray && eltype(T) === Bool)) ||
        throw(ArgumentError(
            "$(name) requires a Bool-shaped type (Bool or AbstractArray{Bool}); " *
            "got T=$(T). Use e.g. Null(x) or Null(typeof(x)) to construct correctly."))
end

# Jacobian eltype `J = Jac(S, T)` (`S` is output type, `T` is input type).
_jacobian_type(S::Type{<: Number}, T::Type{<: Number}) = promote_type(S, T)

# `similar_type` fills in the trailing `L = M * N`, giving the concrete form
# `SMatrix{M, N, F, M * N}` that `_assert_concrete` accepts.
_jacobian_type(::Type{SVector{M, S}}, ::Type{SVector{N, T}}) where {N, M, S, T} =
    similar_type(SMatrix{M, N, S}, promote_type(S, T))

_jacobian_type(S::Type, T::Type) = throw(ArgumentError(
    "differentiate: unsupported shape pair eltype(s)=$S vs eltype(v)=$T. " *
    "Only (Number, Number) and matching-N (SVector{N}, SVector{N}) pairs are " *
    "supported by default. To enable additional combinations, add a method:\n" *
    "    StencilCore._jacobian_type(::Type{S}, ::Type{T}) = <Jacobian type>"))

# Linear-map space of a value-space type `T` — the type whose `one(·)` is the
# multiplicative identity for things in `T`'s algebra. Delegates to
# `_jacobian_type(T, T)` (defined in `differentiate.jl`, resolved at call time):
# Number stays itself; SVector{N, F} maps to the canonical square SMatrix{N, N, F};
# same-type T returns T (e.g. SMatrix is its own identity space).
_unity_space(::Type{T}) where {T} = _jacobian_type(T, T)

#_value_space(::Type{<: Number}) = Bool
#_value_space(::Type{<: SMatrix{N, N, T}}) where {N, T} =
#    similar_type(SVector{N, T}, Bool)
