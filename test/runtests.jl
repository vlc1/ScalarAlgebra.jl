using Test
using StaticArrays
using Static
using ScalarAlgebra
using AlgebraCore  # simplify, materialize, pushforward, differentiate, substitute

@testset "ScalarAlgebra.jl" begin

    @testset "ScalarSym" begin
        # default T = Float64
        sc = @inferred ScalarSym{:x}()
        @test sc isa ScalarSym{:x, Float64}
        @test eltype(sc) === Float64

        # explicit concrete T (Number)
        sc32 = @inferred ScalarSym{:x, Float32}()
        @test sc32 isa ScalarSym{:x, Float32}

        # SVector element type
        sc_v = @inferred ScalarSym{:v, SVector{3, Float64}}()
        @test sc_v isa ScalarSym{:v, SVector{3, Float64}}
        @test eltype(sc_v) === SVector{3, Float64}

        # abstract T → ArgumentError
        @test_throws ArgumentError ScalarSym{:x, AbstractFloat}()
    end

    @testset "ScalarConst" begin
        # infer T from value (Number)
        sc = @inferred ScalarConst(1.0)
        @test sc isa ScalarConst{Float64}
        @test sc.val === 1.0
        @test eltype(sc) === Float64

        # infer T from value (SVector)
        v = SVector(1.0, 2.0, 3.0)
        sc_v = @inferred ScalarConst(v)
        @test sc_v isa ScalarConst{SVector{3, Float64}}
        @test sc_v.val === v

        # explicit T with implicit conversion
        sc_ex = @inferred ScalarConst{Float64}(1)
        @test sc_ex isa ScalarConst{Float64}
        @test sc_ex.val === 1.0

        # idempotent: AbstractScalar passed through unchanged
        inner = ScalarConst(2.0)
        @test ScalarConst(inner) === inner

        # abstract T → ArgumentError
        @test_throws ArgumentError ScalarConst{AbstractFloat}(1.0)
    end

    @testset "ScalarZero" begin
        # direct Bool construction
        sz = @inferred ScalarZero{Bool}()
        @test sz isa ScalarZero{Bool}
        @test eltype(sz) === Bool

        # via Type{Number} → Bool
        sz_t = @inferred ScalarZero(Float64)
        @test sz_t isa ScalarZero{Bool}

        # via Number value → Bool
        sz_v = @inferred ScalarZero(1.0)
        @test sz_v isa ScalarZero{Bool}

        # via SVector value → SVector{N,Bool}
        sz_sv = @inferred ScalarZero(SVector(1.0, 2.0))
        @test sz_sv isa ScalarZero{SVector{2, Bool}}

        # non-Bool-shaped T → ArgumentError
        @test_throws ArgumentError ScalarZero{Float64}()
    end

    @testset "ScalarOne" begin
        # direct Bool construction
        so = @inferred ScalarOne{Bool}()
        @test so isa ScalarOne{Bool}
        @test eltype(so) === Bool

        # via Type{Number} → Bool (unity space of Float64 is Float64, bool-shape → Bool)
        so_t = @inferred ScalarOne(Float64)
        @test so_t isa ScalarOne{Bool}

        # via Number value → Bool
        so_v = @inferred ScalarOne(1.0)
        @test so_v isa ScalarOne{Bool}

        # via SVector value → SMatrix{N,N,Bool,N*N}
        so_sv = @inferred ScalarOne(SVector(1.0, 2.0))
        @test so_sv isa ScalarOne{SMatrix{2, 2, Bool, 4}}

        # SVector{N,F} directly: not Bool-shaped → ArgumentError
        @test_throws ArgumentError ScalarOne{SVector{2, Float64}}()
    end

    @testset "ScalarCall" begin
        x = ScalarSym{:x}()          # Float64
        y = ScalarSym{:y}()          # Float64
        v = ScalarSym{:v, SVector{3, Float64}}()

        # binary, Number + Number
        sc_add = @inferred x + y
        @test sc_add isa ScalarCall
        @test eltype(sc_add) === Float64

        # binary, AbstractScalar op literal (lifts via asscalar)
        sc_mul = @inferred x * 2.0
        @test sc_mul isa ScalarCall
        @test eltype(sc_mul) === Float64

        # unary, Number
        sc_neg = @inferred -x
        @test sc_neg isa ScalarCall
        @test eltype(sc_neg) === Float64

        # unary, SVector element type
        sc_neg_v = @inferred -v
        @test sc_neg_v isa ScalarCall
        @test eltype(sc_neg_v) === SVector{3, Float64}

        # binary, SVector + SVector
        w = ScalarSym{:w, SVector{3, Float64}}()
        sc_add_v = @inferred v + w
        @test sc_add_v isa ScalarCall
        @test eltype(sc_add_v) === SVector{3, Float64}
    end

    @testset "ScalarRef" begin
        # linear indexing with SVector
        v = ScalarSym{:v, SVector{3, Float64}}()
        i = ScalarSym{:i, Int}()
        sc_lin = @inferred v[i]
        @test sc_lin isa ScalarRef
        @test eltype(sc_lin) === Float64

        # linear indexing with Int literal (lifted to ScalarConst)
        sc_lin_lit = @inferred v[1]
        @test sc_lin_lit isa ScalarRef
        @test eltype(sc_lin_lit) === Float64

        # cartesian indexing with SMatrix
        m = ScalarSym{:m, SMatrix{3, 3, Float64, 9}}()
        j = ScalarSym{:j, Int}()
        sc_cart = @inferred m[i, j]
        @test sc_cart isa ScalarRef
        @test eltype(sc_cart) === Float64

        # verify indices are wrapped in tuple
        @test sc_lin.indices isa Tuple{Vararg{AbstractScalar{Int}}}
        @test sc_cart.indices isa Tuple{Vararg{AbstractScalar{Int}}}
    end

    @testset "simplify" begin
        x = ScalarSym{:x}()
        z = ScalarZero(Float64)
        z32 = ScalarZero(Float32)
        c2 = ScalarConst(2.0)
        c3 = ScalarConst(3.0)
        u = ScalarOne(Float64)

        # additive identity: x + 0 = x
        expr1 = x + z
        result1 = @inferred simplify(expr1)
        @test result1 === x

        # additive identity: 0 + x = x
        expr2 = z + x
        result2 = @inferred simplify(expr2)
        @test result2 === x

        # additive identity: 0 + 0 = 0 with promoted type
        expr3 = z + z32
        result3 = @inferred simplify(expr3)
        @test result3 isa ScalarZero
        @test eltype(result3) === Bool

        # constant folding: ScalarConst + ScalarConst
        expr4 = c2 + c3
        result4 = @inferred simplify(expr4)
        @test result4 isa ScalarConst{Float64}
        @test result4.val === 5.0

        # constant folding: ScalarOne + ScalarConst → 1 + const
        expr5 = u + c2
        result5 = @inferred simplify(expr5)
        @test result5 isa ScalarConst{Float64}
        @test result5.val === 3.0

        # constant folding: ScalarConst + ScalarOne → const + 1
        expr6 = c2 + u
        result6 = @inferred simplify(expr6)
        @test result6 isa ScalarConst{Float64}
        @test result6.val === 3.0

        # single-arg addition: +(x) = x
        expr7 = ScalarCall(+, (x,))
        result7 = @inferred simplify(expr7)
        @test result7 === x
    end

    @testset "subtract" begin
        x = ScalarSym{:x}()
        z = ScalarZero(Float64)
        z32 = ScalarZero(Float32)
        c2 = ScalarConst(2.0)
        c3 = ScalarConst(3.0)
        u = ScalarOne(Float64)

        # subtractive identity: x - 0 = x
        expr1 = x - z
        result1 = @inferred simplify(expr1)
        @test result1 === x

        # subtractive identity: 0 - 0 = 0 with promoted type
        expr2 = z - z32
        result2 = @inferred simplify(expr2)
        @test result2 isa ScalarZero
        @test eltype(result2) === Bool

        # negation: 0 - x = -x
        expr3 = z - x
        result3 = @inferred simplify(expr3)
        @test result3 isa ScalarCall
        @test result3.fn === (-)
        @test only(result3.args) === x

        # constant folding: ScalarConst - ScalarConst
        expr4 = c3 - c2
        result4 = @inferred simplify(expr4)
        @test result4 isa ScalarConst{Float64}
        @test result4.val === 1.0

        # constant folding: ScalarOne - ScalarConst
        expr5 = u - c2
        result5 = @inferred simplify(expr5)
        @test result5 isa ScalarConst{Float64}
        @test result5.val === -1.0

        # constant folding: ScalarConst - ScalarOne
        expr6 = c2 - u
        result6 = @inferred simplify(expr6)
        @test result6 isa ScalarConst{Float64}
        @test result6.val === 1.0

        # double negation: -(-x) = x
        neg_x = ScalarCall(-, (x,))
        expr7 = ScalarCall(-, (neg_x,))
        result7 = @inferred simplify(expr7)
        @test result7 === x

        # unary negation: -(x) = -x
        expr8 = -x
        result8 = @inferred simplify(expr8)
        @test result8 isa ScalarCall
        @test result8.fn === (-)
        @test only(result8.args) === x
    end

    @testset "rdivide" begin
        x = ScalarSym{:x}()
        z = ScalarZero(Float64)
        z32 = ScalarZero(Float32)
        c2 = ScalarConst(2.0)
        c3 = ScalarConst(3.0)
        u = ScalarOne(Float64)

        # right-division identity: x / 1 = x
        expr1 = x / u
        result1 = @inferred simplify(expr1)
        @test result1 === x

        # zero-annihilation: 0 / x = 0
        expr2 = z / x
        result2 = @inferred simplify(expr2)
        @test result2 isa ScalarZero
        @test eltype(result2) === Bool

        # constant folding: ScalarConst / ScalarConst
        expr3 = c3 / c2
        result3 = @inferred simplify(expr3)
        @test result3 isa ScalarConst{Float64}
        @test result3.val === 1.5

        # ScalarOne / ScalarConst (identity rule doesn't apply: 1 / x ≠ 1)
        expr4 = u / c2
        result4 = @inferred simplify(expr4)
        @test result4 isa ScalarCall
        @test result4.fn === (/)

        # ScalarConst / ScalarOne (identity: x / 1 = x)
        expr5 = c2 / u
        result5 = @inferred simplify(expr5)
        @test result5 === c2

        # ScalarOne / ScalarOne
        expr6 = u / u
        result6 = @inferred simplify(expr6)
        @test result6 isa ScalarOne
        @test eltype(result6) === Bool

        # ScalarSym / same ScalarSym: x / x = 1
        expr7 = x / x
        result7 = @inferred simplify(expr7)
        @test result7 isa ScalarOne
        @test eltype(result7) === Bool

        # ScalarSym / different ScalarSym: x / y = (x / y) (no simplification)
        y = ScalarSym{:y}()
        expr8 = x / y
        result8 = @inferred simplify(expr8)
        @test result8 isa ScalarCall
        @test result8.fn === (/)
    end

    @testset "ldivide" begin
        x = ScalarSym{:x}()
        z = ScalarZero(Float64)
        z32 = ScalarZero(Float32)
        c2 = ScalarConst(2.0)
        c3 = ScalarConst(3.0)
        u = ScalarOne(Float64)

        # left-division identity: 1 \ x = x
        expr1 = u \ x
        result1 = @inferred simplify(expr1)
        @test result1 === x

        # zero-annihilation: x \ 0 = 0
        expr2 = x \ z
        result2 = @inferred simplify(expr2)
        @test result2 isa ScalarZero
        @test eltype(result2) === Bool

        # constant folding: ScalarConst \ ScalarConst
        expr3 = c2 \ c3
        result3 = @inferred simplify(expr3)
        @test result3 isa ScalarConst{Float64}
        @test result3.val === 1.5

        # ScalarOne \ ScalarConst (identity: 1 \ x = x)
        expr4 = u \ c2
        result4 = @inferred simplify(expr4)
        @test result4 === c2

        # ScalarConst \ ScalarOne (identity rule doesn't apply: x \ 1 ≠ x)
        expr5 = c2 \ u
        result5 = @inferred simplify(expr5)
        @test result5 isa ScalarCall
        @test result5.fn === (\)

        # ScalarOne \ ScalarOne
        expr6 = u \ u
        result6 = @inferred simplify(expr6)
        @test result6 isa ScalarOne
        @test eltype(result6) === Bool

        # ScalarSym \ same ScalarSym: x \ x = 1
        expr7 = x \ x
        result7 = @inferred simplify(expr7)
        @test result7 isa ScalarOne
        @test eltype(result7) === Bool

        # ScalarSym \ different ScalarSym: x \ y = (x \ y) (no simplification)
        y = ScalarSym{:y}()
        expr8 = x \ y
        result8 = @inferred simplify(expr8)
        @test result8 isa ScalarCall
        @test result8.fn === (\)
    end

    @testset "simplify ScalarRef" begin
        @scalar i Int
        @scalar u SVector{2, Float64}
        @scalar v SVector{2, Float64}
        @scalar w SVector{2, Float64}

        # (u + v)[i] → u[i] + v[i]
        r1 = @inferred simplify((u + v)[i])
        @test r1 isa ScalarCall{typeof(+)}
        @test r1.args[1] isa ScalarRef && r1.args[1].arr === u
        @test r1.args[2] isa ScalarRef && r1.args[2].arr === v

        # (u - v)[i] → u[i] - v[i]
        r2 = @inferred simplify((u - v)[i])
        @test r2 isa ScalarCall{typeof(-)}
        @test r2.args[1].arr === u && r2.args[2].arr === v

        # (-u)[i] → -(u[i])
        r3 = @inferred simplify((-u)[i])
        @test r3 isa ScalarCall{typeof(-)}
        @test only(r3.args).arr === u

        # ((u + v) + w)[i] → (u[i] + v[i]) + w[i]  (tests recursion)
        r4 = @inferred simplify(((u + v) + w)[i])
        @test r4 isa ScalarCall{typeof(+)}
        @test r4.args[1] isa ScalarCall{typeof(+)}
        @test r4.args[2] isa ScalarRef && r4.args[2].arr === w

        # ScalarZero[i] — no structural rule, left as ScalarRef
        z = ScalarZero(SVector{2, Float64})
        r5 = @inferred simplify(z[i])
        @test r5 isa ScalarRef
        @test r5.arr === z

        # leaf[i] unchanged (fallback)
        r6 = @inferred simplify(u[i])
        @test r6 isa ScalarRef && r6.arr === u

        # ScalarZero[ScalarConst] → ScalarZero{Bool}
        z_sv = ScalarZero(SVector{2, Float64})
        @test (@inferred simplify(z_sv[ScalarConst(1)])) isa ScalarZero{Bool}
        @test (@inferred simplify(z_sv[ScalarConst(2)])) isa ScalarZero{Bool}
        @test_throws BoundsError simplify(z_sv[ScalarConst(3)])

        # ScalarOne[ScalarConst, ScalarConst] → ScalarConst{Bool} (type-stable)
        o_sv = ScalarOne(SVector{2, Float64})
        r_o1 = @inferred simplify(o_sv[ScalarConst(1), ScalarConst(1)])
        @test r_o1 isa ScalarConst{Bool} && r_o1.val === true
        r_o2 = @inferred simplify(o_sv[ScalarConst(2), ScalarConst(2)])
        @test r_o2 isa ScalarConst{Bool} && r_o2.val === true
        r_o3 = @inferred simplify(o_sv[ScalarConst(1), ScalarConst(2)])
        @test r_o3 isa ScalarConst{Bool} && r_o3.val === false
        r_o4 = @inferred simplify(o_sv[ScalarConst(2), ScalarConst(1)])
        @test r_o4 isa ScalarConst{Bool} && r_o4.val === false
        @test_throws BoundsError simplify(o_sv[ScalarConst(3), ScalarConst(1)])

        # non-pointwise op (* ) does not distribute — result is valid ScalarRef
        # previously threw MethodError: getindex(2, Colon(), 1)
        d = simplify(differentiate(2u, u[1]))
        @test d isa AbstractScalar
        @test eltype(d) === SVector{2, Int64}

        # post-distribution constant fold: (2u)[1] w.r.t. u[1] = 2
        d2 = simplify(differentiate((2u)[1], u[1]))
        @test d2 isa ScalarConst{Int64}
        @test d2.val === 2
    end

@testset "materialize" begin
        @scalar x Float64
        @scalar v SVector{2, Float64}
        @scalar i Int

        @test materialize(x, (x = 3.0,)) === 3.0
        @test materialize(ScalarConst(2.0), NamedTuple()) === 2.0
        @test materialize(ScalarZero(Float64), NamedTuple()) === false
        @test materialize(ScalarOne(Float64), NamedTuple()) === true

        pairs = (x = 2.0,)
        @test materialize(x + ScalarConst(1.0), pairs) === 3.0
        @test materialize(-x, pairs) === -2.0

        pairs_v = (v = SVector(1.0, 2.0), i = 2)
        @test materialize(v[i], pairs_v) === 2.0
    end

    @testset "substitute" begin
        @scalar x Float64
        @scalar y Float64
        @scalar v SVector{2, Float64}
        @scalar i Int

        # value binding → ScalarConst; partial substitution keeps the unbound symbol
        r = substitute(x + y, (x = 1.0,))
        @test r isa ScalarCall{typeof(+)}
        @test r.args[1] === ScalarConst(1.0)
        @test r.args[2] === y
        @test materialize(r, (y = 4.0,)) === materialize(x + y, (x = 1.0, y = 4.0))

        # expression binding splices a subtree
        r2 = substitute(x, (x = y + ScalarConst(1.0),))
        @test r2 isa ScalarCall{typeof(+)}
        @test materialize(r2, (y = 2.0,)) === 3.0

        # AbstractScalar binding passed through via asscalar (idempotent)
        @test substitute(x, (x = y,)) === y

        # leaves untouched; absent symbol kept
        @test substitute(ScalarConst(2.0), (x = 1.0,)) === ScalarConst(2.0)
        @test substitute(ScalarZero(Float64), (x = 1.0,)) isa ScalarZero
        @test substitute(y, (x = 1.0,)) === y

        # array symbol + ScalarRef + index substitution
        rr = substitute(v[i], (v = SVector(5.0, 6.0), i = 2))
        @test rr isa ScalarRef
        @test materialize(rr, NamedTuple()) === 6.0

        # round-trip equivalence with materialize across a small tree
        expr = (2 * x + v[i]) / y
        full = (x = 1.5, v = SVector(3.0, 4.0), i = 1, y = 2.0)
        @test materialize(substitute(expr, full), NamedTuple()) === materialize(expr, full)

        # type stability (each call is a distinct specialization → single branch)
        @test (@inferred substitute(x, (x = 1.0,))) === ScalarConst(1.0)
        @test (@inferred substitute(x, (y = 1.0,))) === x
        @test (@inferred substitute(x + y, (x = 1.0,))) isa ScalarCall
    end

    @testset "differentiate" begin
        x = ScalarSym{:x}()
        y = ScalarSym{:y}()
        @scalar v SVector{2, Float64}
        @scalar w SVector{2, Float64}
        @scalar i Int

        # finite-difference Jacobian (columns = directional derivatives)
        fdj(bf, x0; h = 1e-6) = (n = length(x0);
            hcat([(bf(x0 .+ h .* ((1:n) .== j)) .- bf(x0 .- h .* ((1:n) .== j))) ./ (2h) for j in 1:n]...))

        # self / cross derivatives keep clean structural identity / zero nodes
        @test @inferred(differentiate(x, x)) isa ScalarOne{Bool}
        @test @inferred(differentiate(x, y)) isa ScalarZero{Bool}
        @test @inferred(differentiate(v, v)) isa ScalarOne{SMatrix{2,2,Bool,4}}
        @test @inferred(differentiate(v, x)) isa ScalarZero{SVector{2,Bool}}

        # d(x)/dv: scalar out, vector in → 1×2 zero row
        d_sv = @inferred differentiate(x, v)
        @test eltype(d_sv) === SMatrix{1,2,Bool,2}
        @test materialize(simplify(d_sv), NamedTuple()) == SMatrix{1,2,Bool}(false, false)

        # scalar chain rules, via the JVP core (vs finite differences)
        @test @inferred(differentiate(x + y, x)) isa AbstractScalar
        for (e, jl) in [(x * y, (a, b) -> b), (x / y, (a, b) -> 1 / b), (x \ y, (a, b) -> -b / a^2)]
            @test materialize(simplify(differentiate(e, x)), (x = 2.0, y = 5.0)) ≈ jl(2.0, 5.0)
        end

        # dense Jacobians (vector input) reconstructed from JVP, vs finite differences
        # d(x*v)/dv = x*I
        Jxv = @inferred differentiate(x * v, v)
        @test eltype(Jxv) === SMatrix{2,2,Float64,4}
        @test materialize(simplify(Jxv), (x = 2.5,)) ≈ SMatrix{2,2}(2.5, 0.0, 0.0, 2.5)
        # d(v[1]*w)/dv — the outer-product case: value-space products, no operand swap
        J1w = differentiate(v[ScalarConst(1)] * w, v)
        @test materialize(simplify(J1w), (w = SVector(2.0, 3.0), v = SVector(0.0, 0.0))) ≈
            fdj(vv -> vv[1] .* [2.0, 3.0], [0.0, 0.0])
        # gradient: scalar built from a vector → 1×2 row
        g = differentiate(v[ScalarConst(1)] / v[ScalarConst(2)], v)
        @test eltype(g) === SMatrix{1,2,Float64,2}
        @test materialize(simplify(g), (v = SVector(2.0, 4.0),)) ≈ fdj(vv -> [vv[1] / vv[2]], [2.0, 4.0])
        # column: vector out, scalar in
        c = differentiate(sin(x) * v, x)
        @test eltype(c) === SVector{2,Float64}
        @test materialize(simplify(c), (x = 0.7, v = SVector(1.0, 2.0))) ≈
            (sin(0.7 + 1e-6) .* [1.0, 2.0] .- sin(0.7 - 1e-6) .* [1.0, 2.0]) ./ 2e-6

        # scalar literal × / \ array (must not regress)
        @test eltype(simplify(differentiate(2v, v))) === SMatrix{2,2,Float64,4}
        @test eltype(simplify(differentiate(ScalarConst(2) \ v, v))) === SMatrix{2,2,Float64,4}
        @test eltype(@inferred simplify(differentiate(ScalarConst(2.0) \ x, x))) === Float64

        # differentiate w.r.t. one array element: a single JVP seeded with the unit there
        # ∂(scalar independent of v)/∂v[i] = 0
        @test @inferred(differentiate(x, v[i])) isa ScalarZero{Bool}
        # ∂v/∂v[i] = i-th column of the identity, I[:, i]
        d_v_vi = @inferred differentiate(v, v[i])
        @test d_v_vi isa ScalarRef
        @test d_v_vi.arr isa ScalarOne
        @test d_v_vi.indices[1] isa ScalarConst{Colon}
        @test d_v_vi.indices[2] === i
        @test eltype(d_v_vi) === SVector{2,Bool}
        # d(v[1])/dv: scalar ref w.r.t. vector sym → 1×2 row e_1ᵀ. Purely
        # structural (a 0/1 row, no value scalar to promote with) → Bool-shaped,
        # like differentiate(v, v) === ScalarOne{SMatrix{N,N,Bool}}.
        d_v1_v = differentiate(v[ScalarConst(1)], v)
        @test eltype(d_v1_v) === SMatrix{1,2,Bool,2}
        @test materialize(simplify(d_v1_v), NamedTuple()) == SMatrix{1,2}(true, false)
        # d(2v[1])/d(v[2]) = 0
        @test simplify(differentiate(2v[ScalarConst(1)], v[ScalarConst(2)])) == ScalarConst(0)
    end

    @testset "OneHotScalar" begin
        # construction, eltype, materialize, display — Bool-shaped (structure only)
        oh = @inferred OneHotScalar{3, 2}()
        @test oh isa AbstractScalar{SVector{3, Bool}}
        @test eltype(oh) === SVector{3, Bool}
        @test materialize(oh, NamedTuple()) === SVector(false, true, false)
        @test sprint(show, oh) == "e2"
        @test (@inferred OneHotScalar(SVector{3, Float64}, static(2))) isa OneHotScalar{3, 2}
        @test_throws ArgumentError OneHotScalar{3, 4}()   # K > N

        # type-level index folds structurally (type-stable)
        @test (@inferred simplify(oh[static(2)])) isa ScalarOne{Bool}
        @test (@inferred simplify(oh[static(1)])) isa ScalarZero{Bool}
        @test (@inferred simplify(oh[static(3)])) isa ScalarZero{Bool}
        # runtime index → stable Bool ScalarConst (no identity fold)
        @test (@inferred simplify(oh[2])) === ScalarConst(true)
        @test (@inferred simplify(oh[1])) === ScalarConst(false)
    end

    @testset "sparse dense Jacobian" begin
        x = ScalarSym{:x}()
        @scalar v SVector{3, Float64}

        nodes(s::ScalarCall) = 1 + sum(nodes, s.args; init = 0)
        nodes(s::ScalarRef) = 1 + nodes(s.arr) + sum(nodes, s.indices; init = 0)
        nodes(::AbstractScalar) = 1

        # d(x*v)/dv collapses to the structural diagonal SMatrix(x,O,O, O,x,O, O,O,x)
        s = @inferred simplify(differentiate(x * v, v))
        @test s isa ScalarCall{<:Type{<:SMatrix}}
        @test nodes(s) == 10
        @test count(a -> a === x, s.args) == 3
        @test count(a -> a isa ScalarZero, s.args) == 6
        # values unchanged
        @test materialize(s, (x = 2.5,)) ≈ SMatrix{3,3}(2.5,0.0,0.0, 0.0,2.5,0.0, 0.0,0.0,2.5)

        # self-derivative still the clean identity node
        @test (@inferred differentiate(v, v)) isa ScalarOne{SMatrix{3,3,Bool,9}}
    end

    @testset "static indices" begin
        @scalar a; @scalar b; @scalar c; @scalar d; @scalar u; @scalar x
        @scalar v SVector{3, Float64}
        @scalar w SVector{3, Float64}

        # (B) derivative w.r.t. a static-indexed element folds structurally
        @test (@inferred simplify(differentiate(v[static(1)], v[static(2)]))) isa ScalarZero{Bool}
        @test (@inferred simplify(differentiate(v[static(2)], v[static(2)]))) isa ScalarOne{Bool}
        @test materialize(simplify(differentiate(v[static(1)], v[static(2)])), NamedTuple()) === false
        @test materialize(simplify(differentiate(v[static(2)], v[static(2)])), NamedTuple()) === true
        # vector f → one-hot column
        col = @inferred simplify(differentiate(v, v[static(1)]))
        @test col isa OneHotScalar{3, 1}
        @test materialize(col, NamedTuple()) === SVector(true, false, false)
        # static indices also sparsify the product-rule Jacobian
        @test simplify(differentiate(v[static(1)] * w, v[static(1)])) === w

        # OneHotScalar is a constant leaf: differentiating it again is zero
        # (e.g. ∂/∂x of the constant column ∂v/∂v[1])
        @test (@inferred pushforward(col, x, ScalarOne(Float64))) isa ScalarZero{SVector{3, Bool}}
        @test (@inferred differentiate(differentiate(v, v[static(1)]), x)) isa ScalarZero{SVector{3, Bool}}

        # (A) heterogeneous SA-constructor extraction is type-stable with static indices
        m = SMatrix{2,2}(a, b, c, d)
        @test (@inferred simplify(m[static(1), static(2)])) === c   # col-major [1,2] → arg 3
        @test (@inferred simplify(SVector(u, u, u)[static(2)])) === u

        # regression: runtime indices unchanged (value-carrier; runtime extraction)
        @test simplify(differentiate(v[ScalarConst(1)], v[ScalarConst(2)])) === ScalarConst(false)
        @test simplify(m[1, 2]) === c
    end

    @testset "differentiate nonlinear" begin
        x = ScalarSym{:x}()
        y = ScalarSym{:y}()
        fd(f, x0; h = 1e-6) = (f(x0 + h) - f(x0 - h)) / (2h)
        x0 = 1.3

        # unary chain rules: type-stable, numerically correct vs finite difference
        for (expr, jl) in [
            (exp(x),  exp),  (log(x),  log),  (sin(x), sin),
            (cos(x),  cos),  (tan(x),  tan),  (sqrt(x), sqrt),
            (abs(x),  abs),
        ]
            d = @inferred differentiate(expr, x)
            @test d isa AbstractScalar
            @test materialize(simplify(d), (x = x0,)) ≈ fd(jl, x0) atol = 1e-4
        end

        # sign: structurally zero derivative
        d_sign = @inferred differentiate(sign(x), x)
        @test materialize(simplify(d_sign), (x = x0,)) == 0.0

        # power with constant exponent: d(x^3) = 3x^2
        d_pow = @inferred differentiate(x^ScalarConst(3), x)
        @test materialize(simplify(d_pow), (x = x0,)) ≈ fd(z -> z^3, x0) atol = 1e-4

        # power with ScalarOne exponent: d(x^1) = 1
        d_pow1 = @inferred differentiate(x^ScalarOne(Float64), x)
        @test materialize(simplify(d_pow1), (x = x0,)) ≈ 1.0

        # nested chain across two symbols: d(x*exp(y))/dy = x*exp(y)
        d_chain = differentiate(x * exp(y), y)
        @test materialize(simplify(d_chain), (x = 2.0, y = 0.5)) ≈ 2 * exp(0.5)

        # unsupported / non-differentiable ops fail loudly (not MethodError)
        @test_throws ArgumentError differentiate(min(x, y), x)
        @test_throws ArgumentError differentiate(max(x, y), x)
        @test_throws ArgumentError differentiate(x^y, x)
    end

    @testset "StaticArray constructors" begin
        @scalar u Float64
        @scalar v Float32

        # Bare SVector: N inferred from arg count, T from eltype
        sv = @inferred SVector(u, u)
        @test sv isa ScalarCall
        @test eltype(sv) === SVector{2, Float64}

        # Mixed: concrete arg lifted via asscalar
        sv2 = @inferred SVector(u, 1.0)
        @test sv2 isa ScalarCall
        @test eltype(sv2) === SVector{2, Float64}

        # Mixed eltypes: promotes Float32 + Float64 → Float64
        sv3 = @inferred SVector(u, v)
        @test sv3 isa ScalarCall
        @test eltype(sv3) === SVector{2, Float64}

        # Partially-specified type: SVector{3}
        sv4 = @inferred SVector{3}(u, u, u)
        @test sv4 isa ScalarCall
        @test eltype(sv4) === SVector{3, Float64}

        # SMatrix with explicit size params
        sm = @inferred SMatrix{2,2}(u, u, u, u)
        @test sm isa ScalarCall
        @test eltype(sm) === SMatrix{2,2,Float64,4}

        # materialize round-trips
        @test materialize(sv, (u = 2.0,)) === SVector(2.0, 2.0)
        @test materialize(sv2, (u = 3.0,)) === SVector(3.0, 1.0)
        @test materialize(sm, (u = 1.0,)) === SMatrix{2,2}(1.0, 1.0, 1.0, 1.0)

        # display
        @test sprint(show, sv) == "SVector(u, u)"
        @test sprint(show, sm) == "SMatrix(u, u, u, u)"

        # simplify: generic fallback — passes through with simplified args
        sv_s = @inferred simplify(sv)
        @test sv_s isa ScalarCall
        @test eltype(sv_s) === SVector{2, Float64}

        # simplify: constant folding — all-ScalarConst args evaluate immediately
        sv_c = @inferred simplify(SVector(ScalarConst(1.0), ScalarConst(2.0)))
        @test sv_c === ScalarConst(SVector(1.0, 2.0))

        # simplify: ScalarRef look-ahead — linear index into constructor (homogeneous)
        elem1 = @inferred simplify(sv[1])
        @test elem1 === u

        # simplify: ScalarRef look-ahead — cartesian index into SMatrix (homogeneous)
        sm_u = @inferred simplify(SMatrix{2,2}(u, u, u, u)[1, 2])
        @test sm_u === u

        # simplify: ScalarRef look-ahead — cartesian index, heterogeneous args
        @scalar a Float64; @scalar b Float64; @scalar c Float64; @scalar d Float64
        sm_elem = simplify(SMatrix{2,2}(a, b, c, d)[1, 2])
        @test sm_elem === c   # col-major: [1,2] → linear index 3

        # generic fallback also fixes exp/sin/cos/log
        r = @inferred simplify(exp(u))
        @test r isa ScalarCall{typeof(exp)}
        @test r.fn === exp
    end

end
