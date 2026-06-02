module ScalarAlgebra

using StaticArrays

export AbstractScalar,
       ScalarSym,
       ScalarConst,
       ScalarZero,
       ScalarOne,
       ScalarCall,
       ScalarRef,
       @scalar,
       asscalar,
       simplify

include("utils.jl")
include("types.jl")
include("simplify.jl")
include("display.jl")

end # module
