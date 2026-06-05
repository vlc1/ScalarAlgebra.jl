# AGENTS.md

This file provides guidance to agents when working with code in this
repository.

## Project Overview

**ScalarAlgebra** is a Julia package for symbolic scalar algebra with a
type-level expression tree.

**Type system** (`src/types.jl`): All nodes are `AbstractScalar{T}` where `T`
is the concrete element type (Number or StaticArray). Leaf types:
`ScalarSym{S,T}` (named symbol), `ScalarConst{T}` (literal value),
`ScalarZero{T}` (additive identity, Bool-shaped), `ScalarOne{T}`
(multiplicative identity, Bool-shaped). Interior node: `ScalarCall{F,A,T}`
(applies `fn` to `args`). Index node: `ScalarRef{A,I,T}` (symbolic array
indexing).

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

**Outer-product ordering in chain rules**: The Jacobian of a scalar w.r.t.
`SVector{N}` is a row (`SMatrix{1,N}`). When this row Jacobian `du` appears in
a product with an array value `v::SVector{N}`, operand order is critical:
- `du * v` = `SMatrix{1,N} × SVector{N}` → inner product → `SVector{1}` (wrong
  shape)
- `v * du` = `SVector{N} × SMatrix{1,N}` → outer product → `SMatrix{N,N}`
  (correct)

Two differentiation rules require the outer-product order
(`src/differentiate.jl`):
- `scalar × array` multiplication: use `v * du + u * dv` not `du * v + u * dv`
- `scalar \ array` left-division: use `u \ (dv - (u\v) * du)` not `u \ (dv - du
  * (u\v))`

**`differentiate(a, b::ScalarRef)` index construction**: The method
`differentiate(a::AbstractScalar, b::ScalarRef)` computes
`da/d(b.arr[b.indices])`.  The full index tuple must be
`(_colon_pad(eltype(a))..., b.indices...)`:
- `_colon_pad(eltype(a))` adds colons for `a`'s output dimensions (empty for
  `Number` output, one colon for `SVector` output)
- `b.indices` selects the specific input element Using `a.indices` (wrong
  variable) instead of `b.indices` is a silent bug for non-`ScalarRef` `a`.

**`differentiate(a::ScalarRef, b::ScalarSym)` row-dimension preservation**:
When `eltype(a) <: Number` (scalar output) and `eltype(b) <: AbstractArray`
(vector input), the Jacobian must have shape `SMatrix{1,N}`, not `SVector{N}`.
Julia's indexing collapses the row dimension:
- `M[i, :]` = `SMatrix{M,N}[Int, Colon]` → `SVector{N}` (wrong)
- `M[SA[i], :]` = `SMatrix{M,N}[SVector{1,Int}, Colon]` → `SMatrix{1,N}` (correct)

Integer indices in `a.indices` must be wrapped as `ScalarConst(SVector(val))`
before appending `_colon_pad(eltype(b))`. The helper `_ref_indices(indices,
eltype(a), eltype(b))` dispatches on `(Number, AbstractArray)` to apply
`_wrap_scalar_index` over the index tuple; all other cases pass indices through
unchanged. Forgetting this wrap silently produces `SVector` instead of
`SMatrix{1,N}`, breaking downstream product-rule additions.

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
