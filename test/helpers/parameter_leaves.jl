# Call `f` on each leaf of the parameters `g` and on the leaf at the same path in each of `refs`,
# whatever the depth of the nesting: an architecture that holds a network in each of its layers nests
# one level deeper than a plain chain.
function foreach_parameter(f, g, refs...)
    if g isa AbstractArray || g isa Number
        return f(g, (r for r in refs)...)
    end
    for name in keys(g)
        foreach_parameter(f, g[name], (r[name] for r in refs)...)
    end
    nothing
end
