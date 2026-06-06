using Test
using StaticArrays
using ScalarAlgebra

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

    @testset "differentiate" begin
        x = ScalarSym{:x}()
        y = ScalarSym{:y}()

        # ScalarSym: same symbol → ScalarOne
        @test @inferred(differentiate(x, x)) isa ScalarOne{Bool}

        # ScalarSym: different symbol → ScalarZero
        @test @inferred(differentiate(x, y)) isa ScalarZero{Bool}

        # ScalarCall(+): linearity
        d_add = @inferred differentiate(x + y, x)
        @test d_add isa ScalarCall{typeof(+)}
        @test d_add.args[1] isa ScalarOne{Bool}
        @test d_add.args[2] isa ScalarZero{Bool}

        # ScalarCall(-) binary: linearity
        d_sub = @inferred differentiate(x - y, x)
        @test d_sub isa ScalarCall{typeof(-)}
        @test d_sub.args[1] isa ScalarOne{Bool}
        @test d_sub.args[2] isa ScalarZero{Bool}

        # ScalarCall(*): product rule  d(x*y)/dx = 1*y + x*0 = y
        d_mul = @inferred differentiate(x * y, x)
        @test d_mul isa ScalarCall{typeof(+)}

        # ScalarCall(/): quotient rule  d(x/y)/dx = (1 - (x/y)*0) / y = 1/y
        d_rdiv = @inferred differentiate(x / y, x)
        @test d_rdiv isa ScalarCall{typeof(/)}

        # ScalarCall(\): left-div rule  d(x\y)/dx = x\(0 - 1*(x\y)) = -(x\y)/x
        d_ldiv = @inferred differentiate(x \ y, x)
        @test d_ldiv isa ScalarCall{typeof(\)}

        # ScalarRef: d(v[i])/dv = ∂v/∂v padded with Colon → (I)[i, :]
        @scalar v SVector{2, Float64}
        @scalar w SVector{2, Float64}
        @scalar i Int
        d_ref = @inferred differentiate(v[i], v)
        @test d_ref isa ScalarRef
        @test d_ref.arr isa ScalarOne
        @test d_ref.indices[1] === i
        @test d_ref.indices[2] isa ScalarConst{Colon}

        # ScalarRef: d(v[i])/dw = 0 padded with Colon → (0)[i, :]
        d_ref_zero = @inferred differentiate(v[i], w)
        @test d_ref_zero isa ScalarRef
        @test d_ref_zero.arr isa ScalarZero
        @test d_ref_zero.indices[1] === i
        @test d_ref_zero.indices[2] isa ScalarConst{Colon}

        # cross-shape _jacobian_type: scalar out, vector in → 1×N row
        @test ScalarAlgebra._jacobian_type(Float64, SVector{2,Float64}) === SMatrix{1,2,Float64,2}
        # cross-shape _jacobian_type: vector out, scalar in → M-column
        @test ScalarAlgebra._jacobian_type(SVector{2,Float64}, Float64) === SVector{2,Float64}

        # d(x)/dv: scalar sym w.r.t. vector sym → ScalarZero of row shape
        d_sv = @inferred differentiate(x, v)
        @test d_sv isa ScalarZero{SMatrix{1,2,Bool,2}}

        # d(v)/dx: vector sym w.r.t. scalar sym → ScalarZero of column shape
        d_vs = @inferred differentiate(v, x)
        @test d_vs isa ScalarZero{SVector{2,Bool}}

        # scalar literal × array sym: d(2v)/dv = 2*I  (previously threw)
        d_lit = simplify(differentiate(2v, v))
        @test d_lit isa ScalarCall{typeof(*)}
        @test eltype(d_lit) === SMatrix{2,2,Int64,4}

        # array × scalar literal: d(v*3.0)/dv = I*3.0 (must not regress)
        d_arr_lit = simplify(differentiate(v * ScalarConst(3.0), v))
        @test d_arr_lit isa AbstractScalar

        # scalar literal \ array sym: d(2\v)/dv (previously threw)
        d_ldiv_arr = simplify(differentiate(ScalarConst(2) \ v, v))
        @test d_ldiv_arr isa AbstractScalar
        @test eltype(d_ldiv_arr) === SMatrix{2,2,Float64,4}

        # scalar \ scalar regression
        d_ldiv_s = @inferred simplify(differentiate(ScalarConst(2.0) \ x, x))
        @test d_ldiv_s isa AbstractScalar
        @test eltype(d_ldiv_s) === Float64

        # d(x)/d(v[i]): scalar expr w.r.t. ScalarRef — no colon pad (scalar output)
        d_wrt_ref_s = @inferred differentiate(x, v[i])
        @test d_wrt_ref_s isa ScalarRef
        @test d_wrt_ref_s.arr isa ScalarZero
        @test d_wrt_ref_s.indices === (i,)
        @test eltype(d_wrt_ref_s) === Bool

        # d(v)/d(v[i]): vector expr w.r.t. ScalarRef — Colon prepended (vector output)
        d_wrt_ref_v = @inferred differentiate(v, v[i])
        @test d_wrt_ref_v isa ScalarRef
        @test d_wrt_ref_v.arr isa ScalarOne
        @test d_wrt_ref_v.indices[1] isa ScalarConst{Colon}
        @test d_wrt_ref_v.indices[2] === i
        @test eltype(d_wrt_ref_v) === SVector{2,Bool}

        # d(v[1])/d(v): scalar ref w.r.t. vector sym — constant index wrapped to preserve row shape
        d_scalar_ref_wrt_v = @inferred differentiate(v[ScalarConst(1)], v)
        @test d_scalar_ref_wrt_v isa ScalarRef
        @test eltype(d_scalar_ref_wrt_v) === SMatrix{1,2,Bool,2}

        # d(2v[1])/d(v[2]): exercises the fixed product rule path
        d_2v1_wrt_v2 = @inferred differentiate(2v[ScalarConst(1)], v[ScalarConst(2)])
        @test d_2v1_wrt_v2 isa AbstractScalar
        @test eltype(d_2v1_wrt_v2) === Float64

        # simplify(d(2v[1])/d(v[2])) = 0
        @test simplify(d_2v1_wrt_v2) == ScalarConst(0)
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
