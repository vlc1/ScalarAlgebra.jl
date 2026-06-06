# Scalars render without going through `simplify` (no rewriter at this layer
# yet). Leaves: Symbolic prints its symbol; Constant prints its stored `val`;
# Null and Unity print the `0`/`1` glyphs (type-agnostic, like the term-side
# Zero). Scalar interior nodes render infix when the op is in `_INFIX`, else
# as a call.

const _SCALAR_INFIX = (:+, :-, :*, :/, :\, :^)

Base.show(io::IO, s::AbstractScalar) = _scalar_show(io, s)

_scalar_show(io::IO, ::ScalarSym{S}) where {S} = print(io, S)
_scalar_show(io::IO, s::ScalarConst) = show(io, s.val)
_scalar_show(io::IO, ::ScalarZero) = print(io, 'O')
_scalar_show(io::IO, ::ScalarOne) = print(io, 'U')

_fn_name(fn) = nameof(fn)
_fn_name(::Type{<:SVector})      = :SVector
_fn_name(::Type{<:SMatrix})      = :SMatrix
_fn_name(::Type{<:StaticArray})  = :SArray

function _scalar_show(io::IO, s::ScalarCall)
    op, args = _fn_name(s.fn), s.args

    if length(args) == 2 && op in _SCALAR_INFIX
        print(io, '(')
        _scalar_show(io, args[1])
        print(io, ' ', op, ' ')
        _scalar_show(io, args[2])
        print(io, ')')
    elseif length(args) == 1 && op === :-
        print(io, '-')
        _scalar_show(io, args[1])
    else
        print(io, op, '(')
        for (i, a) in enumerate(args)
            i == 1 || print(io, ", ")
            _scalar_show(io, a)
        end
        print(io, ')')
    end
end

function _scalar_show(io::IO, s::ScalarRef)
    _scalar_show(io, s.arr)
    print(io, '[')
    for (k, idx) in enumerate(s.indices)
        k == 1 || print(io, ", ")
        _scalar_show(io, idx)
    end
    print(io, ']')
end
