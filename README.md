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
| `OneHotScalar{N,K}` | leaf | Bool-shaped structural basis vector `e_K ∈ SVector{N,Bool}` (Jacobian seed) |

**`simplify`** — post-order tree rewrite. Rules cover identity elements,
zero annihilation, constant folding, double negation, same-symbol cancellation,
and index distribution through pointwise nodes. All rules are type-stable
(`@inferred`-safe); `ScalarOne` index results return `ScalarConst` rather than
a union.

**`pushforward`** — the differentiation core: `pushforward(f, x, ẋ)` is the
forward-mode directional derivative (JVP) of `f` along input tangent `ẋ`. The
result lives in `f`'s own value space, so every product in the chain/Leibniz
rules is an ordinary value-space op — correct by construction, no covector or
operand-order bookkeeping. Rules cover `+ - * / \ ^` and the unary nonlinear
functions (`exp, log, sin, cos, tan, sqrt, abs, sign`); `min`/`max` and
non-constant exponents raise a clear `ArgumentError`.

**`differentiate`** — the dense Jacobian, reconstructed from `pushforward` by
seeding the input tangent space: scalar input → one JVP (`ScalarOne` seed);
`SVector{N}` input → one JVP per basis direction, seeded with a structural
`OneHotScalar` and assembled into `SMatrix{M,N}` (vector output) or `SMatrix{1,N}`
(scalar-output gradient row). The one-hot seeds carry their hot position in the
type (via [`Static`](https://github.com/SciML/Static.jl)), so indexing folds to
`ScalarOne`/`ScalarZero` and the Jacobian collapses to its sparse structural form
(`d(x*v)/dv → SMatrix(x,0,0, 0,x,0, 0,0,x)`). Self-derivatives fold to clean
`ScalarOne` identities.

**`materialize`** — evaluates a tree against a `NamedTuple` of bindings.

**`substitute`** — like `materialize` but stays symbolic: replaces the named
symbols by `asscalar(value)` (a raw value becomes a `ScalarConst`; an
`AbstractScalar` is spliced in) and returns an `AbstractScalar`. Partial —
unbound symbols are kept — and lazy (no simplification).

## Verbs live in AlgebraCore

`simplify`, `materialize`, `pushforward`, `differentiate`, and `substitute` are
generic functions owned by
[`AlgebraCore`](https://github.com/vlc1/AlgebraCore.jl); ScalarAlgebra only adds
methods for its types. Bring the verbs into scope with `using AlgebraCore`.

## Quick start

```julia
using AlgebraCore, ScalarAlgebra, StaticArrays

@scalar x Float64
@scalar u SVector{2, Float64}

# Build expressions with standard operators
expr = 2u + x * u

# Differentiate
J = differentiate(expr, u)        # Jacobian w.r.t. u — eltype SMatrix{2,2}
s = simplify(J)                   # structural simplification

# Evaluate
v = materialize(expr, (x = 1.0, u = SVector(3.0, 4.0)))

# Partial substitution stays symbolic
p = substitute(expr, (x = 1.0,))   # 2u + 1.0*u, still an AbstractScalar in u
```

## Bool-shaped identity types

`ScalarZero` and `ScalarOne` encode shape without storing a value. The type
parameter is Bool-shaped: `Bool` for `Number` eltypes, `SMatrix{N,N,Bool}`
for `SVector{N}` eltypes. This lets simplification rules collapse identity
and zero expressions purely by dispatch.

## Static indices

Index with `static(k)` (`using Static`) to keep the index in the type domain.
Then constant folds are fully type-stable and structural where runtime indices
cannot be: `SMatrix(a,b,c,d)[static(1),static(2)]` extracts `c` type-stably even
for heterogeneous constructors, and `differentiate(v[static(i)], v[static(j)])`
folds to a structural `ScalarOne`/`ScalarZero` (Kronecker δ) — which also
sparsifies product-rule Jacobians like `d(v[static(1)]*w)/d(v[static(1)])`.
Runtime indices (`v[1]`) still work but yield value-carrying `ScalarConst`s.

## To do

- Reverse-mode pullback / VJP for efficient scalar-output gradients.
- Matrix-valued (`SMatrix`) symbol differentiation.
