# `materialize` is owned by AlgebraCore; these methods extend it for AbstractScalar.
# Evaluate a scalar expression tree by substituting symbol bindings from `pairs`.
materialize(sc::ScalarSym{S}, pairs::NamedTuple) where {S} = pairs[S]
materialize(sc::ScalarConst, ::NamedTuple) = sc.val
materialize(::ScalarZero{T}, ::NamedTuple) where {T} = zero(T)
materialize(::ScalarOne{T}, ::NamedTuple) where {T} = one(T)
materialize(::OneHotScalar{N, K}, ::NamedTuple) where {N, K} =
    SVector{N, Bool}(ntuple(m -> m == K, N))
materialize(sc::ScalarCall, pairs::NamedTuple) =
    sc.fn(map(a -> materialize(a, pairs), sc.args)...)
materialize(sc::ScalarRef, pairs::NamedTuple) =
    materialize(sc.arr, pairs)[map(i -> materialize(i, pairs), sc.indices)...]
#
#Base.convert(::Type{Expr}, sc::ScalarSym{S}) where {S} = :($S::$(eltype(sc)))
#Base.convert(::Type{Expr}, sc::ScalarConst) = :($(sc.val)::$(eltype(sc)))
#Base.convert(::Type{Expr}, ::ScalarZero{T}) where {T} = :(zero($T))
#Base.convert(::Type{Expr}, ::ScalarOne{T}) where {T} = :(one($T))
#Base.convert(::Type{Expr}, sc::ScalarCall) =
#    Expr(:(::), Expr(:call, sc.fn, convert.(Expr, sc.args)...), eltype(sc))
#Base.convert(::Type{Expr}, sc::ScalarRef) =
#    Expr(:(::), Expr(:ref, convert(Expr, sc.arr), convert.(Expr, sc.indices)...), eltype(sc))
