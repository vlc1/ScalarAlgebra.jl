module ScalarAlgebra

using StaticArrays
using Static

# The algebra verbs are owned by AlgebraCore; ScalarAlgebra extends them.
import AlgebraCore: simplify,
                    substitute,
                    materialize,
                    pushforward,
                    differentiate

export AbstractScalar,
       ScalarSym,
       ScalarConst,
       ScalarZero,
       ScalarOne,
       ScalarCall,
       ScalarRef,
       OneHotScalar,
       @scalar,
       asscalar

include("utils.jl")
include("types.jl")
include("simplify.jl")
include("materialize.jl")
include("substitute.jl")
include("pushforward.jl")
include("differentiate.jl")
include("display.jl")

end # module
