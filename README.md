# ScalarAlgebra

## Types

| Type | Kind | `T` constraint | Role |
|------|------|----------------|------|
| `ScalarSym{S,T}` | leaf | concrete | Named symbol |
| `ScalarConst{T}` | leaf | concrete | Literal value carrier |
| `ScalarZero{T}` | leaf | Bool-shaped | Additive identity |
| `ScalarOne{T}` | leaf | Bool-shaped | Multiplicative identity |
| `ScalarCall{F,A,T}` | interior | `promote_op` | Applies `fn::F` to `args::A` |
| `ScalarRef{A,I,T}` | interior | `eltype(eltype(A))` | Symbolic array index |

**Bool-shaped**: `ScalarZero` and `ScalarOne` do not store a value. Their type
parameter encodes the shape of zero/identity: `Bool` for `Number` eltypes,
`SMatrix{N,N,Bool,N²}` for `SVector{N}` eltypes (the square identity matrix
shape).

## To-do list

- `simplify`: `pow`
- `substitute`: remove `ScalarSym` without checking type-matching and recompute
  tree
- `materialize`: assume all `ScalarSym` have been flushed out
- `differentiate`
