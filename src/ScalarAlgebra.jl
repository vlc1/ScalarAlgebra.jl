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
       simplify,
       materialize,
       differentiate

include("utils.jl")
include("types.jl")
include("simplify.jl")
include("materialize.jl")
include("differentiate.jl")
include("display.jl")

end # module
