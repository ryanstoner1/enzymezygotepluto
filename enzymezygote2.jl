### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 3ec71c12-a31a-417a-a018-4908a39010ae
using Zygote

# ╔═╡ 9f73b406-de7e-491f-ad04-0bd279e20362
using FillArrays

# ╔═╡ 82260001-6aa8-46e4-9e98-d30425ca541b
using Enzyme

# ╔═╡ b46b486f-9893-41b2-8252-6614fcaab5ec
using BenchmarkTools

# ╔═╡ 84b3b3dd-a5a9-4d6a-85fd-8fe5c9e95cc1
md"""
# Enzyme pullbacks in Zygote (Part 2/3)
"""

# ╔═╡ e144f64c-893b-4442-9abb-d3e50a45b658
md"""
# 
**Recap**: In **Part 1** I showed that ``Zygote`` can't do mutation for arrays but hinted that ``Enzyme`` could. 

As a geoscientist, the ultimate wish list for many programs dealing with geospatial data is to:

###### 1. Allow mutation of arrays/matrices
###### 2. Allow BLAS operations
###### 3. Be fast (say within 5x of the forward pass)
###### 4. Be easy to implement  

---

``Enzyme``, however, has its own drawbacks. It can't perform most BLAS operations (matrix multiplication last time I checked). So we need ``Zygote`` for requirement **2**.

"""

# ╔═╡ 80bcfad6-1432-4872-951e-fe527bc70d04
md"""
---
### Game Plan: Enzyme + Zygote
In this **Part 2** I'll show how ``Enzyme`` pullbacks can be incorporated into ``Zygote``, which will mostly take care of *requirements* **1** and **2**. Requirement **3** could be improved upon, but it's better than the first cut in **Part 1**. 

Overall, this approach should circumvent many of the limitations of ``Enzyme`` and ``Zygote``. 

In the following part, **Part 3** I'll show how a hackish, but functional way to automate this this process and help with requirement **4**.

---
### Steps:
###### 1. Isolate mutating part of code in own function
###### 2. Implement ``Enzyme`` pullback
###### 3. Put pullback in custom `Zygote.@adjoint`
###### 4. Benchmark it all
---

We'll be using the same example as in **Part 1**:
"""


# ╔═╡ 67420d36-ac70-4981-9164-ac21e8818d5e
begin
	n = 3
	a = zeros(n,n) # preallocate
	b = rand(n,n)
	c = rand(n)
end

# ╔═╡ 0079604a-a06b-412b-8375-0f57e06f9434


# ╔═╡ acc217d1-8f28-43ce-9b47-629924c2d5c9
for i in 1:n
	val = sum(b\c)
	a[i,i] = val
	c[i] = sin(val)
end

# ╔═╡ e0035373-dea5-41a1-a7de-361324b9cd4d
md"""
The first idea is to **place the block where mutation happens in its own function**. Let's call it `f1!()`
"""

# ╔═╡ 88f35c6e-c261-4d46-8859-5a9e631d6713
function f1!(a,c,val,i)
	a[i,i] = val
	c[i] = sin(val)
    return nothing
end

# ╔═╡ d1beccb5-20c1-42ca-942e-18d45439a5d0
md"""

I couldn't find a way to make everything work without defining a nonmutating version for the `Zygote.@adjoint`. Unfortunately, this does reallocate values, but for me it ends up being simpler to implement than staying purely in ``Zygote``. 

As we'll see later, it ends up being fairly fast.
"""

# ╔═╡ 27c82c72-1970-449e-9c4d-494ec3df5315
function f1(a,c,val,i)
	f1!(a,c,val,i)
 	return a,c
end

# ╔═╡ 67c19da5-be1b-487e-ba28-e41170c66557
md"""
##### Now for the pullback . . .

``Zygote`` sometimes uses `Fill` values from `FillArrays` to presumably decrease memory requirements. ``Enzyme`` doesn't know about these and will squeal if they aren't imported. 

Most of this modified from the ``Enzyme`` pullback guide [here](https://docs.juliahub.com/Enzyme/G1p5n/0.6.2/pullbacks/).

"""

# ╔═╡ bd88e905-e7c6-47f2-890b-d4d18d147332
Zygote.@adjoint function f1(a,c,val,i)
	f1!(a,c,val,i)
	function backf1(inmutvars)
		(∂z_a,∂z_c)=inmutvars 
		∂z_a = ∂z_a isa Fill ? collect(∂z_a) : ∂z_a # protect Enzyme from `Fills`
		∂z_c = ∂z_c isa Fill ? collect(∂z_c) : ∂z_c		
        ∂z_a = ∂z_a≡nothing ? zero(a) : ∂z_a # protect Enzyme from `nothing`s
		∂z_c = ∂z_c≡nothing ? zero(c) : ∂z_c

		(∂z_val,)=Enzyme.autodiff(
			f1!,Const,Duplicated(a,∂z_a),
			Duplicated(c,∂z_c),
			Active(val),
			i)::Tuple{Float64,}

		return ∂z_a,∂z_c,∂z_val,0.0
	end
	return (a,c),backf1
 end

# ╔═╡ e5794cd4-7109-49d5-801c-186d915f8ce4
md"""
One of the rough patches is dealing with `nothing`s from ``Zygote``. Ideally these would be passed through, but I substitute them with 0s.
"""

# ╔═╡ 01ce4adc-6ad1-494a-9005-6990008ec754
md"""

---
Putting now let's place the example code in a function and use `f1()`:
"""

# ╔═╡ ab236f88-e1f7-4a05-937b-9f8264ffaaeb
function hybrid(b,n)
	a = zeros(n,n) # preallocate
	c = rand(n)
	for i in 1:n
		val = sum(b\c)
		a,c = f1(a,c,val,i)
	end
	return sum(a)
end

# ╔═╡ 2e2098cd-e5bd-426f-853e-edda332626da
md"""

Let's test the autodiff capabilities in ``Zygote``.

"""

# ╔═╡ cdda2127-7f63-43a5-be6a-5780966c0834
try
	Zygote.gradient(hybrid,b,n)
catch y
	@warn "Exception: $y !"
end

# ╔═╡ fe6f5cf0-8246-4ab9-b49c-69ffdfdc5f1a
md"""
### Now, some benchmarking!
---

Let's check out the original function: 

"""

# ╔═╡ 011f4100-bf65-4444-8948-906814698b0e
@benchmark hybrid($b,$n)

# ╔═╡ e388bf57-b777-42db-a5f3-75f121302c55
@benchmark Zygote.gradient($hybrid,$b,$n)

# ╔═╡ 38aafdac-844a-425a-9e58-cfa7ccb038c2
md"""

This is faster than *Part 1* that was just in Zygote.

Increasing `n` makes the performance difference is less dramatic. 
"""

# ╔═╡ fedb1ffb-7297-414e-ae90-529685ccc3bb
begin 
	n4 = n*4
	b4 = rand(n4,n4)
end

# ╔═╡ 3560246a-b09f-40a1-a86d-e68e70fc8a49
@benchmark hybrid($b4,$n4)

# ╔═╡ d8b5223a-25ae-48db-bf4f-724bec4e5026
@benchmark Zygote.gradient($hybrid,$b4,$n4)

# ╔═╡ 78c3ab6d-1d42-493d-a467-fdf1ed4e13b7
md"""
#### Implementing in a larger codebase?

Code performance is one thing, but human performance is something else. Often preexisting code will have *dozens* of locations where mutation takes place. 

Custom-writing pullbacks would could be prohibitively times consuming to implement. This is why I wrote **Part 3** with a temporary workaround until we can mutate arrays in ``Zygote`` of consistently use ``BLAS`` in ``Enzyme``. 

"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
BenchmarkTools = "~1.3.1"
Enzyme = "~0.10.1"
FillArrays = "~0.13.2"
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

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

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

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

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

[[deps.Enzyme]]
deps = ["Adapt", "CEnum", "Enzyme_jll", "GPUCompiler", "LLVM", "Libdl", "LinearAlgebra", "ObjectFile", "Printf", "Random"]
git-tree-sha1 = "6b65e97271ac8de8ffcef0f7ba17ec065e9cc6f5"
uuid = "7da242da-08ed-463a-9acd-ee780be4f1d9"
version = "0.10.1"

[[deps.Enzyme_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg", "TOML"]
git-tree-sha1 = "722aa3b554e883118e0e3111629ec40e176cee2c"
uuid = "7cc45869-7501-5eee-bdea-0790c847d4ef"
version = "0.0.33+0"

[[deps.ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

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

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "21b5d9da260afa6a8638ba2aaa0edbbb671c37bd"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.16.0"

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

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "e7e9184b0bf0158ac4e4aa9daf00041b5909bf1a"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.14.0"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg", "TOML"]
git-tree-sha1 = "771bfe376249626d3ca12bcd58ba243d3f961576"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.16+0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

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

[[deps.ObjectFile]]
deps = ["Reexport", "StructIO"]
git-tree-sha1 = "55ce61d43409b1fb0279d1781bf3b0f22c83ab3b"
uuid = "d8793406-e978-5875-9003-1fc021f44a92"
version = "0.3.7"

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

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

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

[[deps.StructIO]]
deps = ["Test"]
git-tree-sha1 = "010dc73c7146869c042b49adcdb6bf528c12e859"
uuid = "53d494c1-5632-5724-8f4c-31dff12d585f"
version = "0.3.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "464d64b2510a25e6efe410e7edab14fffdc333df"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.20"

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
# ╟─84b3b3dd-a5a9-4d6a-85fd-8fe5c9e95cc1
# ╟─e144f64c-893b-4442-9abb-d3e50a45b658
# ╠═3ec71c12-a31a-417a-a018-4908a39010ae
# ╟─80bcfad6-1432-4872-951e-fe527bc70d04
# ╠═67420d36-ac70-4981-9164-ac21e8818d5e
# ╠═0079604a-a06b-412b-8375-0f57e06f9434
# ╠═acc217d1-8f28-43ce-9b47-629924c2d5c9
# ╟─e0035373-dea5-41a1-a7de-361324b9cd4d
# ╠═88f35c6e-c261-4d46-8859-5a9e631d6713
# ╟─d1beccb5-20c1-42ca-942e-18d45439a5d0
# ╠═27c82c72-1970-449e-9c4d-494ec3df5315
# ╟─67c19da5-be1b-487e-ba28-e41170c66557
# ╠═9f73b406-de7e-491f-ad04-0bd279e20362
# ╠═82260001-6aa8-46e4-9e98-d30425ca541b
# ╠═bd88e905-e7c6-47f2-890b-d4d18d147332
# ╟─e5794cd4-7109-49d5-801c-186d915f8ce4
# ╟─01ce4adc-6ad1-494a-9005-6990008ec754
# ╠═ab236f88-e1f7-4a05-937b-9f8264ffaaeb
# ╟─2e2098cd-e5bd-426f-853e-edda332626da
# ╠═cdda2127-7f63-43a5-be6a-5780966c0834
# ╟─fe6f5cf0-8246-4ab9-b49c-69ffdfdc5f1a
# ╠═b46b486f-9893-41b2-8252-6614fcaab5ec
# ╠═011f4100-bf65-4444-8948-906814698b0e
# ╠═e388bf57-b777-42db-a5f3-75f121302c55
# ╟─38aafdac-844a-425a-9e58-cfa7ccb038c2
# ╠═fedb1ffb-7297-414e-ae90-529685ccc3bb
# ╠═3560246a-b09f-40a1-a86d-e68e70fc8a49
# ╠═d8b5223a-25ae-48db-bf4f-724bec4e5026
# ╟─78c3ab6d-1d42-493d-a467-fdf1ed4e13b7
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
