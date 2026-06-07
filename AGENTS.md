# AGENTS.md

This file provides guidance to agents when working with code in this
repository.

## Project Overview

**ScalarAlgebra** is a Julia package for symbolic scalar algebra with a
type-level expression tree.

**Verbs owned by AlgebraCore**: the generic functions `simplify`, `substitute`,
`materialize`, `pushforward`, `differentiate` are declared (with their generic
docstrings) in the `AlgebraCore` dependency. ScalarAlgebra `import`s and
**extends** them; it does **not** export them (users `using AlgebraCore`). When
adding/changing a verb method, define it on the imported function — do not
redeclare `function <verb> end` here.

**`substitute`** (`src/substitute.jl`): like `materialize` but stays in the
algebra — replaces each `ScalarSym` named in `pairs` by `asscalar(pairs[name])`
(raw value → `ScalarConst`; `AbstractScalar` → spliced as-is), keeps unbound
symbols, and rebuilds nodes structurally (no simplification). Leaves use the
`substitute(::AbstractScalar, ::NamedTuple) = sc` fallback; `ScalarSym`,
`ScalarCall`, `ScalarRef` specialize (the latter two via the recursive
`_substitute_args`).

**Type system** (`src/types.jl`): All nodes are `AbstractScalar{T}` where `T`
is the concrete element type (Number or StaticArray). Leaf types:
`ScalarSym{S,T}` (named symbol), `ScalarConst{T}` (literal value),
`ScalarZero{T}` (additive identity, Bool-shaped), `ScalarOne{T}`
(multiplicative identity, Bool-shaped). Interior node: `ScalarCall{F,A,T}`
(applies `fn` to `args`). Index node: `ScalarRef{A,I,T}` (symbolic array
indexing). Seed leaf: `OneHotScalar{N,K}` (Bool-shaped structural basis vector `e_K ∈
SVector{N,Bool}`, hot position `K` in the type; value eltype comes from
promotion, like `ScalarZero`/`ScalarOne`).

**Bool-shaped identity types**: `ScalarZero` and `ScalarOne` carry a
Bool-shaped type parameter — Bool for scalar eltypes, SMatrix{N,N,Bool} for
SVector eltypes — to represent structural zero/identity without embedding a
runtime value. Construction accepts the operand eltype and auto-normalizes via
`_to_bool_shape` / `_unity_space` (`src/utils.jl`).

**Simplification** (`src/simplify.jl`): `simplify` traverses the tree
post-order via `simplify(::ScalarCall)` → `_simplify_args` (recurse children) →
`_simplify_call(fn, args)`. Each operator dispatches to a family of
`_simplify_<op>` methods that match on argument types. Rules cover: identity
elements (add/sub/mul/div), zero-annihilation, constant folding (ScalarConst
and ScalarOne), double negation, and same-symbol cancellation (ScalarSym /
ScalarSym = 1). Shape guards (`Base.promote_op`) prevent identity collapse when
the result type would change. `ScalarRef` simplification (`_simplify_ref`)
distributes indexing down through `ScalarCall` nodes and folds constant indices
into `ScalarConst`, `ScalarZero`, and `ScalarOne` leaves.

**Type stability**: All `simplify` and `differentiate` rules must return a
single concrete type — never `Union{A, B}`. Use `@inferred` in every new test.
When a rule would naturally return different node types based on a runtime
value (e.g. diagonal vs off-diagonal in `ScalarOne`), return a value-carrying
node (`ScalarConst`) instead of a structurally-typed node to preserve type
stability.

**Differentiation = forward-mode pushforward** (`src/differentiate.jl`): the
core primitive is `pushforward(f, x::ScalarSym, ẋ)` — the directional derivative
(JVP) of `f` along input tangent `ẋ`. Its result lives in `f`'s **own value
space** (`eltype(pushforward(f, …)) == eltype(f)`): the tangent of a
scalar-valued node is a scalar, of a vector-valued node a vector. Consequently
**every product in the chain/Leibniz rules is an ordinary value-space op** —
there is no covector, no `SMatrix{1,N}` row, and no operand-order bookkeeping.
The rules dispatch via `_pushforward_call(fn, args, x, ẋ)` (mirroring
`_simplify_call`); `*` is uniformly `u*pf(v) + pf(u)*v`.

The dense Jacobian `differentiate(f, x)` is **reconstructed from `pushforward`**
by seeding the input tangent space (`src/differentiate.jl`):
- scalar input (`eltype(x)<:Number`): one JVP seeded with `ScalarOne` →
  scalar/column output by `eltype(f)`.
- vector input (`SVector{N}`): one JVP per basis direction, seeded with a
  **structural** `OneHotScalar{N,K,T}` (`_basis`); columns built by the
  `@generated` `_jvp_columns` (literal `K`, so each column type is concrete).
  `_assemble_jacobian` is the **single place** output shape is materialized —
  `SMatrix{M,N}` (vector output) or `SMatrix{1,N}` (scalar-output row), built via
  the SA-constructor interception. It is `@generated` for `SVector` output so
  rows index with `StaticInt`; then `OneHotScalar[StaticInt]` folds to
  `ScalarOne`/`ScalarZero` (`_simplify_ref`, `src/simplify.jl`) and the identity
  rules collapse the Jacobian to its **sparse** structural form (e.g.
  `d(x*v)/dv → SMatrix(x,O,O, O,x,O, O,O,x)`). Type-level indices (via `Static`)
  are required: a runtime index would make the One/Zero choice a `Union`.
- `differentiate(f, b::ScalarRef)` = `∂f/∂(b.arr[b.indices])`: one JVP seeded by
  `_unit_seed`. A **static** index (`b.indices :: ScalarConst{StaticInt}`) seeds
  with a structural `OneHotScalar{N,K}` so the result folds to `ScalarOne`/
  `ScalarZero` (Kronecker δ); a runtime/symbolic index uses the `I[:, k]` slice
  (`ScalarRef(ScalarOne, (Colon, k))`) and folds to a Bool `ScalarConst`.

Static user indices in general (`v[static(k)]`, `StaticInt` in the type) make
constant folds type-stable where runtime indices cannot: the `@generated`
`_simplify_ref_call(::Type{SA}, args, ::Tuple{Vararg{ScalarConst{<:StaticInt}}})`
(`src/simplify.jl`) extracts from a heterogeneous SA-constructor type-stably,
and the static `_unit_seed`/`OneHotScalar` folds above. Runtime indices keep the
existing (Union-prone-but-functional) paths.

Self-derivatives keep clean structural nodes: scalar self folds to
`ScalarOne{Bool}` via the seed; vector self has a dedicated more-specific method
returning `ScalarOne(SVector{N})` (no ambiguity with the generic vector method).
There is **no** `_jacobian_type`/`_colon_pad`/`_ref_indices` machinery anymore —
shape is handled only in `_assemble_jacobian`. `_unity_space` (`src/utils.jl`)
survives solely for `ScalarOne` construction.

**`_simplify_ref` distribution rules**: Index distribution through `ScalarCall`
nodes must be op-specific (`src/simplify.jl`). The dispatch pattern is
`_simplify_ref(call::ScalarCall, indices)` → `_simplify_ref_call(call.fn,
call.args, indices)`.

Rules by operator:
- `+`, `-`: distribute to all args; `Number`-eltype args are scalar broadcast
  factors — keep them as-is, only distribute into array-eltype args
- `*`: `scalar × array` → keep scalar, distribute index into array; `array ×
  scalar` → distribute into array, keep scalar; `array × array` → fallback
(`ScalarRef(call, indices)`)
- `/`: `array / scalar` → distribute into numerator only; otherwise fallback
- `\`: `scalar \ array` → distribute into dividend only; otherwise fallback
- all others: fallback

**Post-distribution constant folding**: After `_simplify_ref` distributes an
index into leaf nodes (`ScalarConst`, `ScalarOne`, `ScalarZero`), the resulting
sub-expression is often immediately foldable. Always use `_simplify_call(fn,
args)` — not the bare `ScalarCall(fn, args)` constructor — when assembling the
distributed result. `ScalarCall(fn, args)` bypasses all simplification rules,
leaving `2 * true` unsimplified instead of `ScalarConst(2)`.

## MCP Servers

### julia-mcp (REQUIRED for Julia work)

Always use julia-mcp for executing Julia code. Do NOT use Bash `julia`
commands.

julia-mcp provides:
- Persistent REPL session state across multiple code evaluations
- Efficient package management and compilation caching
- Better integration with the development environment
- Access to interactive Julia development

Use `mcp__julia__julia_eval` to run Julia code. This maintains state, avoids
repeated compilation, and is the standard Julia development tool for this
project.

**Stale session detection**: If `Pkg.test` passes in a subprocess but direct
`julia_eval` reproduces old behavior after editing source files, the session
has stale compiled code. Call `mcp__julia__julia_restart` (with `env_path`)
before re-running interactive checks. Revise.jl is loaded automatically but
cannot always hot-patch method table changes (e.g., adding a new dispatch
method that shadows a previously compiled one).
