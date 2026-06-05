# ScalarAlgebra

Symbolic scalar algebra for Julia with a fully static, type-level expression
tree. Supports `Number` and `StaticArrays` eltypes throughout.

## Features

**Expression tree** — every node is `AbstractScalar{T}` where `T` is the
concrete element type computed at construction time via `Base.promote_op`.

| Type | Kind | Role |
|------|------|------|
| `ScalarSym{S,T}` | leaf | named symbol, materialized by name lookup |
| `ScalarConst{T}` | leaf | literal value carrier (`Number`, `SArray`, `Colon`, …) |
| `ScalarZero{T}` | leaf | structural additive identity (Bool-shaped `T`) |
| `ScalarOne{T}` | leaf | structural multiplicative identity (Bool-shaped `T`) |
| `ScalarCall{F,A,T}` | node | applies `fn::F` to `args::A` |
| `ScalarRef{A,I,T}` | node | symbolic array index `arr[indices...]` |

**`simplify`** — post-order tree rewrite. Rules cover identity elements,
zero annihilation, constant folding, double negation, same-symbol cancellation,
and index distribution through pointwise nodes. All rules are type-stable
(`@inferred`-safe); `ScalarOne` index results return `ScalarConst` rather than
a union.

**`differentiate`** — symbolic Jacobian. Chain rules for `+`, `-`, `*`, `/`,
`\`, and `ScalarRef`. Jacobian shape follows `_jacobian_type(S, T)`:
`(Number, Number) → promote_type`, `(SVector{M}, SVector{N}) → SMatrix{M,N}`,
`(Number, SVector{N}) → SMatrix{1,N}`, `(SVector{M}, Number) → SVector{M}`.
Scalar-times-array and scalar-left-divides-array branches use the outer-product
operand order to preserve shape.

**`materialize`** — evaluates a tree against a `NamedTuple` of bindings.

## Quick start

```julia
using ScalarAlgebra, StaticArrays

@scalar x Float64
@scalar u SVector{2, Float64}

# Build expressions with standard operators
expr = 2u + x * u

# Differentiate
J = differentiate(expr, u)        # Jacobian w.r.t. u — eltype SMatrix{2,2}
s = simplify(J)                   # structural simplification

# Evaluate
v = materialize(expr, (x = 1.0, u = SVector(3.0, 4.0)))
```

## Bool-shaped identity types

`ScalarZero` and `ScalarOne` encode shape without storing a value. The type
parameter is Bool-shaped: `Bool` for `Number` eltypes, `SMatrix{N,N,Bool}`
for `SVector{N}` eltypes. This lets simplification rules collapse identity
and zero expressions purely by dispatch.
