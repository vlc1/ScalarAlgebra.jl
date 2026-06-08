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

# Multiplicative-identity (unity) space of a value-space type `T` — the type
# whose `one(·)` is the multiplicative identity for things in `T`'s algebra:
# Number stays itself; SVector{N,F} maps to the canonical square SMatrix{N,N,F}.
# Used by `ScalarOne` construction (`src/types.jl`).
_unity_space(::Type{T}) where {T <: Number} = T
_unity_space(::Type{SVector{N, T}}) where {N, T} = similar_type(SMatrix{N, N, T}, T)
_unity_space(T::Type) = throw(ArgumentError(
    "ScalarOne: unsupported value-space type $T. Only Number and SVector{N} are " *
    "supported (matrix-valued symbols are out of scope)."))

# Resolve the fully-specified StaticArray type from a (possibly bare/partial) SA type
# and a tuple of AbstractScalar args whose eltypes determine T.
function _scalar_sa_type(::Type{SA}, args::Tuple) where {SA <: StaticArray}
    T = promote_type(map(eltype, args)...)
    similar_type(SA, T)
end

# Bare SVector (no size parameter): infer N from arg count.
function _scalar_sa_type(::Type{<:SVector}, args::Tuple)
    T = promote_type(map(eltype, args)...)
    SVector{length(args), T}
end
