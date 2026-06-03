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

end
