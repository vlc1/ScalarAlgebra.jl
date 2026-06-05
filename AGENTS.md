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
the result type would change.

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
