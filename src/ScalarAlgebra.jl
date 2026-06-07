module ScalarAlgebra

using StaticArrays
using Static

export AbstractScalar,
       ScalarSym,
       ScalarConst,
       ScalarZero,
       ScalarOne,
       ScalarCall,
       ScalarRef,
       OneHotScalar,
       @scalar,
       asscalar,
       simplify,
       materialize,
       pushforward,
       differentiate

include("utils.jl")
include("types.jl")
include("simplify.jl")
include("materialize.jl")
include("pushforward.jl")
include("differentiate.jl")
include("display.jl")

end # module
