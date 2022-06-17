### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ f5d7445a-453b-4504-9b44-63bb581e1bed
using Zygote

# ╔═╡ 1d2840fc-9125-49e4-b4e6-60979d1a717c
using Random

# ╔═╡ a03be71d-3c19-4505-bbd1-f0d9457b9fab
using LinearAlgebra

# ╔═╡ 6551ea8f-8cbc-44f5-945d-84214d1ac1a6
using BenchmarkTools

# ╔═╡ b902b1fa-ed98-11ec-2ae5-d1fc8b5f626c
md"""
# Enzyme pullbacks in Zygote (Part 1/3)
"""

# ╔═╡ facfb625-87e1-41a3-b811-5f2effe160d3
md"""

**Background**: 
This article presupposes some knowledge of autodiff and `Zygote` and, of course, `Julia`. 

--- 

"""

# ╔═╡ f87217bd-7aa6-4179-9979-f9228d09bd1b
md"""
First bring some packages we'll use.
"""

# ╔═╡ f79c6a9d-a2ca-460d-b726-4f0473f42b78
md"""
# BLAS and Mutation: The Autodiff Challenge for Geoscience
"""

# ╔═╡ 12377278-b45e-42ff-9deb-873d0b64cbd0
md"""
Codes in the geosciences almost always capture fluxes between different domains over time and space. For this reason, geospatial data is almost always represented as matrices. These matrices are then often operated upon with linear algebra, which in `Julia` uses the BLAS library. 
"""

# ╔═╡ 9f351910-2309-4ef7-b5f3-01cb27a0c51f
md"""
Therefore, any autodiff package that would work with the geoscientific data would ideally:

###### 1. Allow mutation of arrays/matrices
###### 2. Allow BLAS operations
###### 3. Be fast (say within 5x of the forward pass)
###### 4. Be easy to implement  

---
*Turns out this isn't easy in ``Julia's`` most popular autodiff packages currently.*
**Part 1** of this post shows why it is hard to fulfill even two of these four requirements for types of code geoscientists might use in `Zygote`. 
**Part 2** shows how `Enzyme` pullbacks can be integrated into `Zygote.`
**Part 3** shows a hacky, but functional workaround that solved 3.5/4 of the four issues for me.  

As usual, *please* let me know about any errors or ways to improve the code. 

"""


# ╔═╡ 4d760fdf-4c38-4a7b-a906-4b4093c55f9b
md"""
---
**First** I'll give a **toy example** with matrices that might represent some geospatial data.
"""

# ╔═╡ d4bf1b62-c45f-46e8-ab11-929af6909ab7
begin
	n = 3
	a = zeros(n,n) # preallocate
	b = rand(n,n)
	c = rand(n)
end

# ╔═╡ 34d7b188-2204-4522-8517-187dd9fa424f
md"""
Now let's populate the diagonals of variable `a` with the output of some linear algebra operation and modify `c` to approximate nonlinear iteration. 
"""

# ╔═╡ 63218e10-8b2e-44ab-8fef-d52fb023f00d
for i in 1:n
	val = sum(b\c)
	a[i,i] = val
	c[i] = sin(val)
end

# ╔═╡ c64d5d90-4d89-4388-9e48-bc59816b75ec
md"""
`a` will then be the variable of interest
"""

# ╔═╡ da9c5be8-bb1f-4f55-a5e9-f0a3d8a420e7
md"""
# Zygote: The Challenge of Mutation
"""

# ╔═╡ 85c19224-136a-4aef-a467-f8b3999829f1
md"""
Ideally we would simply place the code above in a function and pass that to `Zygote.gradient().` Let's try that assuming that the matrix `b` and `n` are inputs.
"""

# ╔═╡ cf012a26-bf47-4491-97f1-d1896ba04bba
function sad(b,n)
	a = zeros(n,n)
	c = rand(n)
	for i in 1:n
		val = sum(b\c)
		a[i,i] = val
		c[i] = sin(val)
	end
	return sum(a)
end

# ╔═╡ bcdcbd22-33b3-4210-9771-607945a61504
try
	Zygote.gradient(sad,b,n)
catch y
	@warn "Zygote is sad: $y !"
end

# ╔═╡ e94c32b5-2b3e-4317-a45a-069c89c58aa2
md"""
### Mutation isn't possible for most cases in Zygote (for arrays)

See the discussion [here](https://github.com/FluxML/Zygote.jl/pull/75) and [here](https://github.com/FluxML/Zygote.jl/issues/1228) for much better explanations than I could give for why this is.

An alternative is to use the Zygote-provided `Buffer` type if `c` weren't mutated. If this were to work we would hypothetically replace `a = zeros(n,n)` with a `Buffer.` `copy()` is necessary because `Buffers` can't participate in linear algebra operations. However, once `copy()` is called on a buffer it is frozen.
"""

# ╔═╡ 3533f8fb-078b-4cc7-8592-090b33686f12
function mut(b,n)
	a = Zygote.Buffer(zeros(n,n))
	c = Zygote.Buffer(zeros(n))

	for i = 1:n
		c[i] += rand(1)[1]
		val = sum(b\copy(c))
		a[i,i] = val
		c[i] = sin(val)	
	end
	
	return sum(copy(a))
end

# ╔═╡ 3dab11c4-2224-4d60-a74f-9f7723328607
try
	Zygote.gradient(mut,b,n)
catch y
	@warn "Exception: $y !"
end

# ╔═╡ ad6715c5-d223-4c02-98d5-29b53cbfb205
md"""
Removing the `c` mutation makes this run, and some people might find this already is a workable solution for their code structure.
"""

# ╔═╡ 27ac6b0d-fc41-4053-9c58-ce7b294c151a
function nocmut(b,n)
	a = Zygote.Buffer(zeros(n,n))
	c = rand(n)
	for i = 1:n
		val = sum(b\c)
		a[i,i] = val
	end
	
	return sum(copy(a))
end

# ╔═╡ fffd3e9e-001d-4f03-9245-5f900466d7e4
try
	Zygote.gradient(nocmut,b,n)
catch y
	@warn "Exception: $y !"
end

# ╔═╡ da9a9fbc-0f02-4a68-8e26-49bef90592fc
md"""
## What if we throw out mutation?

Although this goes against our original goal it may be worthwhile sometimes. Here's a first pass trying to preserve the structure of the original. I would guess something a similar solution be devised with list comprehension.:
"""

# ╔═╡ 4ed38ff4-5037-41f0-80a3-e13ac54ae965
function nomut(b,n)

	c = rand(n)
	i = 1:n

	# do first loop
	adiag = Float64[]
	cnew = Float64[]
	for i = 1:n
	 	val = sum(b\c)
	 	adiag = vcat(adiag,val)
		if i!=n
			cnew = vcat(cnew,sin(val))
			c = vcat(cnew,c[i+1:end])
		end
	end
	a = diagm(adiag)
	
	return sum(a)
end
	

# ╔═╡ e55947b4-3711-48fd-8b9f-5343d54ec915
try
	Zygote.gradient(nomut,b,n)
catch y
	@warn "Exception: $y !"
end

# ╔═╡ f5afbf28-f4b8-4ec9-96a1-91166bb8809f
md"""
##### This works, but . . .
There is the mutation problem. How does this compare for speed? Let's run some benchmarks.
"""

# ╔═╡ 93c8f283-b55b-4336-ae7f-2ea4a7b88c21
@benchmark Zygote.gradient($nomut,$b,$n)

# ╔═╡ dcc4ec2f-a032-4c84-a955-380f6b3cdc33
md"""
How does this compare to the original performance without AD?
"""

# ╔═╡ da9fd6cc-1b52-4024-96c4-7154c9b13a77
@benchmark nomut($b,$n)

# ╔═╡ c0ebfe98-6220-4bce-81c1-7f3ca2d6ff25
md"""

The performance is ok, but not ideal. Type stability has been usually hard to get for me and preallocation is not meaningful without mutation.

"""

# ╔═╡ 4176644d-7b11-4608-9747-006b7b2b1574
@code_warntype Zygote.gradient(nomut,b,n)

# ╔═╡ a178e563-6798-4ac9-8a1f-1a7704bf601e
md"""
## Issues with Zygote

Honestly, I would be fine with the performance hit. Often it's possible to use `map()`, which is a bit simpler, but still the **real challenge is the implementation**. 

I'll have a hard time convincing other researchers to use `Julia's` autodiff packages if they have to rewrite each mutation call since many codes invoke mutation *dozens* if not *hundreds* of times. In my experience, people will be more likely to stick with their own code and hand-code in adjoints from symbolic differentiation.

"""


# ╔═╡ 493b50a6-5bdd-4fdd-9b68-08dae9d7d86e
md"""
---
In **Part 2** I show how mutation is possible in Enzyme but most BLAS operations are not, and in **Part 3** I show a hacky solution to get the best of both worlds. 
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
BenchmarkTools = "~1.3.1"
Zygote = "~0.6.40"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "6f1d9bc1c08f9f4a8fa92e3ea3cb50153a1b40d4"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.1.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "4c10eee4af024676200bc7752e536f858c6b8f93"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.1"

[[deps.ChainRules]]
deps = ["ChainRulesCore", "Compat", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics"]
git-tree-sha1 = "34e265b1b0049896430625ce1638b2719c783c6b"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.35.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9489214b993cd42d17f44c36e359bf6a7c919abf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "924cdca592bc16f14d2f7006754a621735280b74"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.1.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "28d605d9a0ac17118fe2c5e9ce0fbb76c3ceb120"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "2f18915445b248731ec5db4e4a17e451020bf21e"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.30"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "af14a478780ca78d5eb9908b263023096c2b9d64"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.6"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "09e4b894ce6a976c354a69041a04748180d43637"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.15"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "a9e798cae4867e3a41cae2dd9eb60c047f1212db"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.6"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "2bbd9f2e40afd197a1379aef05e0d85dba649951"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.7"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "IRTools", "InteractiveUtils", "LinearAlgebra", "MacroTools", "NaNMath", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "a49267a2e5f113c7afe93843deea7461c0f6b206"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.40"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─b902b1fa-ed98-11ec-2ae5-d1fc8b5f626c
# ╟─facfb625-87e1-41a3-b811-5f2effe160d3
# ╟─f87217bd-7aa6-4179-9979-f9228d09bd1b
# ╠═f5d7445a-453b-4504-9b44-63bb581e1bed
# ╠═1d2840fc-9125-49e4-b4e6-60979d1a717c
# ╟─f79c6a9d-a2ca-460d-b726-4f0473f42b78
# ╟─12377278-b45e-42ff-9deb-873d0b64cbd0
# ╟─9f351910-2309-4ef7-b5f3-01cb27a0c51f
# ╟─4d760fdf-4c38-4a7b-a906-4b4093c55f9b
# ╠═d4bf1b62-c45f-46e8-ab11-929af6909ab7
# ╟─34d7b188-2204-4522-8517-187dd9fa424f
# ╠═63218e10-8b2e-44ab-8fef-d52fb023f00d
# ╟─c64d5d90-4d89-4388-9e48-bc59816b75ec
# ╟─da9c5be8-bb1f-4f55-a5e9-f0a3d8a420e7
# ╟─85c19224-136a-4aef-a467-f8b3999829f1
# ╠═cf012a26-bf47-4491-97f1-d1896ba04bba
# ╠═bcdcbd22-33b3-4210-9771-607945a61504
# ╟─e94c32b5-2b3e-4317-a45a-069c89c58aa2
# ╠═3533f8fb-078b-4cc7-8592-090b33686f12
# ╠═3dab11c4-2224-4d60-a74f-9f7723328607
# ╟─ad6715c5-d223-4c02-98d5-29b53cbfb205
# ╠═27ac6b0d-fc41-4053-9c58-ce7b294c151a
# ╠═fffd3e9e-001d-4f03-9245-5f900466d7e4
# ╟─da9a9fbc-0f02-4a68-8e26-49bef90592fc
# ╠═a03be71d-3c19-4505-bbd1-f0d9457b9fab
# ╠═4ed38ff4-5037-41f0-80a3-e13ac54ae965
# ╠═e55947b4-3711-48fd-8b9f-5343d54ec915
# ╟─f5afbf28-f4b8-4ec9-96a1-91166bb8809f
# ╠═6551ea8f-8cbc-44f5-945d-84214d1ac1a6
# ╠═93c8f283-b55b-4336-ae7f-2ea4a7b88c21
# ╟─dcc4ec2f-a032-4c84-a955-380f6b3cdc33
# ╠═da9fd6cc-1b52-4024-96c4-7154c9b13a77
# ╟─c0ebfe98-6220-4bce-81c1-7f3ca2d6ff25
# ╠═4176644d-7b11-4608-9747-006b7b2b1574
# ╟─a178e563-6798-4ac9-8a1f-1a7704bf601e
# ╟─493b50a6-5bdd-4fdd-9b68-08dae9d7d86e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
