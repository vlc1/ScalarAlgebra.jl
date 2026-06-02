module ScalarAlgebra

using StaticArrays

export AbstractScalar,
       ScalarSym,
       ScalarConst,
       ScalarZero,
       ScalarOne,
       ScalarCall,
       ScalarRef,
       @scalar

include("utils.jl")
include("types.jl")
include("display.jl")

end # module
