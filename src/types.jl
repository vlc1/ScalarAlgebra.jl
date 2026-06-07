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
`T` is any concrete type — `Number`, `SArray`, `Colon`, etc. — making
`ScalarConst` the right carrier for boundary literals such as `x + 1`, `v +
SVector(1)`, and slice indices such as `ScalarConst(Colon())` (`:`) in
`ScalarRef` index tuples.
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

ScalarZero(::Type{<: AbstractScalar{T}}) where {T} = ScalarZero(T)
ScalarZero(::T) where {T <: AbstractScalar} = ScalarZero(T)

"""
    ScalarOne{T}()
    ScalarOne(val) / ScalarOne(::Type)

Type-level multiplicative identity/structural one for [`AbstractScalar`](@ref).

**Inner constructor** `ScalarOne{T}()`: `T` must be Bool-shaped (Bool for
scalars, SMatrix{N,N,Bool,N²} for arrays). Materializes to `one(T)` at the
actual element type level.

**Outer constructor** `ScalarOne(val)` or `ScalarOne(::Type)`: accepts the
operand's element type (e.g., Float64 or SVector{3,Float64}), promotes to unity
space via `_unity_space`, converts to Bool shape via `_to_bool_shape`, and
calls the inner constructor. For example, `ScalarOne(SVector{2, Float64})` →
`ScalarOne{SMatrix{2,2,Bool,4}}()`.

Lets the scalar `simplify` rules collapse multiplicative identities by dispatch
— structurally, with no `.val` inspection — mirroring how `ScalarZero`
collapses additive identities.
"""
struct ScalarOne{T} <: AbstractScalar{T}
    function ScalarOne{T}() where {T}
        applicable(one, T) || throw(ArgumentError(
            "ScalarOne{T} requires `one(T)` to be defined (a square-scalar shape); got T=$T"))
        S = _to_bool_shape(T)
        _assert_concrete(:ScalarOne, S)
        new{S}()
    end
end

ScalarOne(::Type{T}) where {T} = ScalarOne{_to_bool_shape(_unity_space(T))}()
ScalarOne(::T) where {T} = ScalarOne{_to_bool_shape(_unity_space(T))}()

ScalarOne(::Type{<: AbstractScalar{T}}) where {T} = ScalarOne(T)
ScalarOne(::T) where {T <: AbstractScalar} = ScalarOne(T)

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

# Constructor-typed fn: result type is SA itself — no promote_op needed.
function ScalarCall(fn::Type{SA}, args::A) where {
        SA <: StaticArray, A <: Tuple{Vararg{AbstractScalar}}}
    ScalarCall{Type{SA}, A, SA}(fn, args)
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

# Intercept StaticArray constructors when the first arg is an AbstractScalar.
# Union{AbstractScalar, Any} collapses to Any in Julia's type system, so we
# check the first arg at runtime. Remaining args are lifted via asscalar so
# mixed calls (e.g. SVector(u, 1.0)) are handled transparently. Gap:
# non-scalar-first mixed calls (SVector(1.0, u)) fall through to StaticArrays;
# use ScalarConst explicitly in that case.
function (::Type{SA})(x::AbstractScalar, xs...) where {SA <: StaticArray}
    args = (x, map(asscalar, xs)...)
    ScalarCall(_scalar_sa_type(SA, args), args)
end

"""
    ScalarRef{A, I, T}(arr, indices)

Index node: represents `arr[indices...]` symbolically, mirroring Julia's
`Expr(:ref, arr, i1, ...)` AST node. `arr` is an array-valued scalar;
`indices` is a tuple of integer-valued scalars. The result element type `T`
is the element type of the array type carried by `arr`.
"""
struct ScalarRef{A <: AbstractScalar{<:AbstractArray}, I, T} <: AbstractScalar{T}
    arr::A
    indices::I

    function ScalarRef{A, I, T}(arr::A, indices::I) where {A <: AbstractScalar{<:AbstractArray}, I, T}
        _assert_concrete(:ScalarRef, T)
        new{A, I, T}(arr, indices)
    end
end

function ScalarRef(arr::A, indices::I) where {A <: AbstractScalar{<:AbstractArray}, I <: Tuple}
    T = Base.promote_op(getindex, eltype(A), map(eltype, indices)...)
    ScalarRef{A, I, T}(arr, indices)
end

# IndexLinear: flat integer into any N-D array eltype.
Base.getindex(arr::AbstractScalar{<: AbstractArray}, i) =
    ScalarRef(arr, (asscalar(i),))

# IndexCartesian: exactly N integer indices for an N-D array eltype.
Base.getindex(arr::AbstractScalar{<:AbstractArray{T, N}}, I::Vararg{Any, N}) where {T, N} =
    ScalarRef(arr, map(asscalar, I))

"""
    OneHotScalar{N, K}()
    OneHotScalar(::Type{SVector{N, T}}, ::StaticInt{K})

Bool-shaped structural one-hot basis vector `e_K ∈ SVector{N, Bool}` (a `1` at
position `K`, zeros elsewhere). Like [`ScalarZero`](@ref)/[`ScalarOne`](@ref) it
carries *structure*, not value: it materializes to `SVector{N, Bool}`, and the
value element type comes from promotion at use sites (`x * e_K` is
`SVector{N, Float64}`). The hot position `K` lives in the type, so indexing folds
to a `ScalarOne`/`ScalarZero` **structurally** (see `_simplify_ref`), which lets
dense Jacobians reconstructed from `pushforward` collapse to their sparse form
instead of leaving dead `x*0.0`/`x*1.0` products. Seeded into `differentiate` via
`_jvp_columns`.

The outer constructor accepts the operand value-space type `SVector{N, T}` and
normalizes to the Bool shape, mirroring `ScalarZero(::Type)`.
"""
struct OneHotScalar{N, K} <: AbstractScalar{SVector{N, Bool}}

    function OneHotScalar{N, K}() where {N, K}
        (N isa Int && K isa Int && 1 <= K <= N) || throw(ArgumentError(
            "OneHotScalar{N,K}: need integer literals with 1 ≤ K ≤ N; got N=$(N), K=$(K)"))
        new{N, K}()
    end
end

OneHotScalar(::Type{SVector{N, T}}, ::StaticInt{K}) where {N, T, K} = OneHotScalar{N, K}()
