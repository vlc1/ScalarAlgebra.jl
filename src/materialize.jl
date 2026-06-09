# `isliteral` and `materialize` are owned by AlgebraCore; these methods extend
# them for AbstractScalar.

# An expression is literal (materializable) iff it contains no free symbols.
# Inferred entirely from types.
isliteral(sc::AbstractScalar) = isliteral(typeof(sc))
isliteral(::Type{<:AbstractScalar}) = true
isliteral(::Type{<:ScalarSym}) = false
@generated isliteral(::Type{<:ScalarCall{F, A}}) where {F, A} =
    all(isliteral, A.parameters)
@generated isliteral(::Type{<:ScalarRef{A, I}}) where {A, I} =
    isliteral(A) && all(p -> !(p <: AbstractScalar) || isliteral(p), I.parameters)

# Evaluate a literal scalar expression tree to a concrete value.
materialize(sc::ScalarConst) = sc.val
materialize(::ScalarZero{T}) where {T} = zero(T)
materialize(::ScalarOne{T}) where {T} = one(T)
materialize(::OneHotScalar{N, K}) where {N, K} =
    SVector{N, Bool}(ntuple(m -> m == K, N))
materialize(sc::ScalarCall) = sc.fn(map(materialize, sc.args)...)
materialize(sc::ScalarRef) =
    materialize(sc.arr)[map(materialize, sc.indices)...]
#
#Base.convert(::Type{Expr}, sc::ScalarSym{S}) where {S} = :($S::$(eltype(sc)))
#Base.convert(::Type{Expr}, sc::ScalarConst) = :($(sc.val)::$(eltype(sc)))
#Base.convert(::Type{Expr}, ::ScalarZero{T}) where {T} = :(zero($T))
#Base.convert(::Type{Expr}, ::ScalarOne{T}) where {T} = :(one($T))
#Base.convert(::Type{Expr}, sc::ScalarCall) =
#    Expr(:(::), Expr(:call, sc.fn, convert.(Expr, sc.args)...), eltype(sc))
#Base.convert(::Type{Expr}, sc::ScalarRef) =
#    Expr(:(::), Expr(:ref, convert(Expr, sc.arr), convert.(Expr, sc.indices)...), eltype(sc))
