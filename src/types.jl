"""
    AbstractScalar{T}

Supertype for cell-level scalar expressions. Reaches a single value of
type `T` at `materialize` time (no axes).
"""
abstract type AbstractScalar{T} end

Base.eltype(::Type{<:AbstractScalar{T}}) where {T} = T
Base.eltype(sc::AbstractScalar) = eltype(typeof(sc))

"""
    ScalarSym{S, T}()

Named, runtime-substituted scalar parameter `S` (a `Symbol`) of concrete type
`T` (default `Float64`). Materializes to the value supplied at the keyword `S`
of the `pairs` NamedTuple.
"""
struct ScalarSym{S, T} <: AbstractScalar{T}

    function ScalarSym{S, T}() where {S, T}
        _assert_concrete(:ScalarSym, T)
        new{S, T}()
    end
end

ScalarSym{S}() where {S} = ScalarSym{S, Float64}()

"""
    @scalar name [T = Float64]

Bind `name` to `ScalarSym{:name, T}()`. `@scalar x` ≡
`x = ScalarSym{:x, Float64}()`; `@scalar x Float32` ≡
`x = ScalarSym{:x, Float32}()`.
"""
macro scalar(name, T = :Float64)
    name isa Symbol ||
        throw(ArgumentError("@scalar expects a variable name, got `$(name)`"))
    :($(esc(name)) = $ScalarSym{$(QuoteNode(name)), $(esc(T))}())
end

"""
    ScalarConst{T}(val) / ScalarConst(val)

Literal scalar leaf carrying a value `val::T` as-is. Materializes to `val`.
`T` is any concrete type — `Number`, `SArray`, etc. — making `ScalarConst` the
right carrier for boundary literals such as `x + 1`, `v + SVector(1)`.
"""
struct ScalarConst{T} <: AbstractScalar{T}
    val::T

    function ScalarConst{T}(val) where {T}
        _assert_concrete(:ScalarConst, T)
        new{T}(convert(T, val))
    end
end

ScalarConst(val::T) where {T} = ScalarConst{T}(val)

# Idempotent lift: scalars are already in the algebra.
ScalarConst(sc::AbstractScalar) = sc

"""
    ScalarZero{T}()

Type-level additive identity/structural zero for [`AbstractScalar`](@ref).
Materializes to `zero(T)`; lets the scalar `simplify` and `differentiate`
rules collapse by dispatch.
"""
struct ScalarZero{T} <: AbstractScalar{T}

    function ScalarZero{T}() where {T}
        _assert_bool_shape(:ScalarZero, T)
        new{T}()
    end
end

ScalarZero(T::Type) = ScalarZero{_to_bool_shape(T)}()
ScalarZero(::T) where {T} = ScalarZero{_to_bool_shape(T)}()

"""
    ScalarOne{T}()

Type-level multiplicative identity/structural one for [`AbstractScalar`](@ref).
Materializes to `one(T)`; requires `T` to be a *square scalar* — a type with
`one(T)` defined (`Number`, square `SMatrix{N,N,F}`, ...). Construction rejects
`T` lacking `one(T)` (e.g. `SVector`, non-square `SMatrix`).

Lets the scalar `simplify` rules collapse multiplicative identities by
dispatch — structurally, with no `.val` inspection — mirroring how `ScalarZero`
collapses additive identities.
"""
struct ScalarOne{T} <: AbstractScalar{T}
    function ScalarOne{T}() where {T}
        _assert_bool_shape(:ScalarOne, T)
        applicable(one, T) || throw(ArgumentError(
            "ScalarOne{T} requires `one(T)` to be defined (a square-scalar shape); got T=$T"))
        new{T}()
    end
end

ScalarOne(::Type{T}) where {T} = ScalarOne{_to_bool_shape(_unity_space(T))}()
ScalarOne(::T) where {T} = ScalarOne{_to_bool_shape(_unity_space(T))}()

"""
    ScalarCall(fn, args::Tuple{Vararg{AbstractScalar}})

Interior node of a scalar-tree: applies `fn` to scalar `args` component-wise.
The element type `T = Base.promote_op(fn, eltype.(args)...)` is computed
**at construction**; a `Union{}` result throws (the node is unconstructable).
"""
struct ScalarCall{F, A<:Tuple{Vararg{AbstractScalar}}, T} <: AbstractScalar{T}
    fn::F
    args::A

    ScalarCall{F, A, T}(fn::F, args::A) where {F, A<:Tuple{Vararg{AbstractScalar}}, T} =
        new{F, A, T}(fn, args)
end

function ScalarCall(fn::F, args::A) where {F, A<:Tuple{Vararg{AbstractScalar}}}
    T = Base.promote_op(fn, map(eltype, args)...)
    T === Union{} && throw(ArgumentError(
        "unconstructable ScalarCall: $(fn) over eltypes $(map(eltype, args)) has " *
        "no result type (Base.promote_op returned Union{})"))
    ScalarCall{F, A, T}(fn, args)
end

# Promote a non-AbstractScalar value to a scalar leaf at the operator boundary.
# Wraps as `ScalarConst` — a literal carrier, no `one(·)` multiplication.
asscalar(sc::AbstractScalar) = sc
asscalar(x) = ScalarConst(x)

Base.convert(::Type{<:AbstractScalar}, x) = ScalarConst(x)

# Unary operator overloads
for op in (:-, :+, :exp, :sin, :cos, :tan, :log, :sqrt, :abs, :sign)
    @eval Base.$op(a::AbstractScalar) = ScalarCall($op, (a,))
end

# Binary operator overloads
# Every binary op with at least one AbstractScalar lifts into a `ScalarCall` tree.
for op in (:+, :-, :*, :/, :\, :^, :min, :max)
    @eval Base.$op(a::AbstractScalar, b::AbstractScalar) = ScalarCall($op, (a, b))
    @eval Base.$op(a::AbstractScalar, b) = ScalarCall($op, (a, asscalar(b)))
    @eval Base.$op(a, b::AbstractScalar) = ScalarCall($op, (asscalar(a), b))
end

"""
    ScalarRef{A, I, T}(arr, indices)

Index node: represents `arr[indices...]` symbolically, mirroring Julia's
`Expr(:ref, arr, i1, ...)` AST node. `arr` is an array-valued scalar;
`indices` is a tuple of integer-valued scalars. The result element type `T`
is the element type of the array type carried by `arr`.
"""
struct ScalarRef{
        A <: AbstractScalar{<:AbstractArray},
        I <: Tuple{Vararg{AbstractScalar{Int}}},
        T} <: AbstractScalar{T}
    arr::A
    indices::I

    ScalarRef{A, I, T}(arr::A, indices::I) where {
            A <: AbstractScalar{<:AbstractArray},
            I <: Tuple{Vararg{AbstractScalar{Int}}},
            T} = new{A, I, T}(arr, indices)
end

function ScalarRef(arr::A, indices::I) where {
        A <: AbstractScalar{<:AbstractArray},
        I <: Tuple{Vararg{AbstractScalar{Int}}}}
    T = eltype(eltype(A))
    ScalarRef{A, I, T}(arr, indices)
end

# indexing overloads
const IntLike = Union{AbstractScalar{Int}, Int}

# IndexLinear: flat integer into any N-D array eltype.
Base.getindex(arr::AbstractScalar{<:AbstractArray}, i::IntLike) =
    ScalarRef(arr, (asscalar(i),))

# IndexCartesian: exactly N integer indices for an N-D array eltype.
Base.getindex(arr::AbstractScalar{<:AbstractArray{T,N}}, I::Vararg{IntLike,N}) where {T,N} =
    ScalarRef(arr, map(asscalar, I))
