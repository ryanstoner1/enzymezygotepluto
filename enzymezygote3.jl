### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 50346be3-9633-41cc-a929-b66865a5cb0f
using Random

# ╔═╡ 446832dd-f75e-41f3-8ce3-a38e58c87126
using CodeTracking

# ╔═╡ 80901e1a-8088-4802-aaef-a6d3276a9004
md"""
# Enzyme pullbacks in Zygote (Part 3/3)
"""

# ╔═╡ 1f4a9328-8da1-4ffc-ab0d-97739ec546c1
md"""
## Recap

This is a series about autodiff in `Julia`. 
In **Part 1** I showed how the package ``Zygote`` can't mutate arrays.

In **Part 2** I introduced ``Enzyme`` and how it and ``Zygote`` can complement each other. 

Specifically, ``Enzyme`` allows for mutations, but cannot perform most BLAS operations, which are the default for linear algebra. ``Zygote`` can, so my solution was to isolate blocks where mutation happens and write a custom `Zygote.@adjoint` for them.

There were four requirements that would be nice to have for geoscience in `Julia's` autodiff system.

###### 1. Allow mutation of arrays/matrices
###### 2. Allow BLAS operations
###### 3. Be fast (say within 5x of the forward pass)
###### 4. Be easy to implement  
--- 

**This section will be about the 4th requirement.**

--- 

## Easy Implementation?

In this section I run through a program to generate pullback code from **Part 2**. We'll use the same example code as the previous two Parts:

"""

# ╔═╡ 69a5f7a4-9be2-47e6-92af-a3b633497a08
begin 
	n = 3
	b = rand(n,n)
	a = zeros(n,n)
	c = rand(n)
end

# ╔═╡ 86653499-902e-40d1-9bb8-602e8aa21082
md"""

Before we had:
	``` 
for i in 1:n
	 	val = sum(b\c)
	 	a[i,i] = val
	 	c[i] = sin(val)
end 
```
Again, we will isolate the mutating part into a function, but try to figure out what the arguments should be to then populate the pullback in ``Enzyme``. 
"""

# ╔═╡ 6327deea-8e21-4bc6-9e25-16ddb056e5ce
for i in 1:n
		val = sum(b\c)
		function f1()
			a[i,i] = val
			c[i] = sin(val)
		end
end

# ╔═╡ 21064167-a0da-4e9e-9d9a-6244720fc6b5
md"""
Then let's set up some arrays to store lists of variables to construct the Enzyme call.
"""

# ╔═╡ 8f20995c-9813-45be-9342-faca9ad614b8
funcs = []

# ╔═╡ 8f14239b-85e0-496c-8fcb-ae5251ab1260
varsmutarr = []

# ╔═╡ 09f76434-e162-46a0-986c-291e76aaf805
varsarr= []

# ╔═╡ 8e34f06e-a247-4bcf-9903-2fb7672fc96e
md"""

Next we'll grab the variables currently used and transform the function into a string we can perform regex on. To do that I'll use Tim Holly's excellent `CodeTracking` package. It doesn't work in `Pluto` because of the way scoping works, but uncommenting the `@code_string` line should work for you.

"""

# ╔═╡ b55fa59a-a81e-4b7b-aa26-6d88e6552812
md"""
After the function is processed then save all of the variable names in the arrays we set up.
"""

# ╔═╡ 57ea9c88-ca05-4e31-a099-96a0072c15f4
for i in 1:n
	val = sum(b\c)    
	
	# get list of local and global variables defined up till f1() runs
	localf1 = string.(keys(Base.@locals)) # string array of local variables
	globf1 = getindex.(getfield(varinfo().content[1],1),1) # string array of 			global variables
	varf1 = union(globf1,localf1)
	varf1mut = [var*r"\[" for var in varf1] # build list of parameters to check 		for mutation w. regex
	function f1()
		a[i,i] = val
		c[i] = sin(val)
	end

	f1()

	# Because of scoping issues this won't work in Pluto.jl, but it should work outside of pluto
	# f1str = @code_string f1() # make function a string to do regex on
	f1str = "function f1()
		a[i,i] = val
		c[i] = sin(val)
	end"
	
 	f1str = split(f1str,"\n")[2:end-1] # chop off function definition and "end"
 	
	f1str = f1str.*"\n"
 	f1str = *(f1str...)
	if varf1 ∉ varsarr
		push!(varsarr,varf1)
		push!(varsmutarr,varf1mut)
		push!(funcs,f1str)
	end 
end

# ╔═╡ 84b4c6dd-3b2d-4b67-aa67-383e0791a506
md"""
#### Phew!

It looks complicated, but if we see what's in the `vars` array, then it becomes more clear.

"""

# ╔═╡ 81c0a936-6f11-4c0d-bc90-810bc52c4aca
varsarr[1] # names of all variables seen before f1() is run

# ╔═╡ 7670b446-a993-4d63-ba54-7e74690d1a52
funcs[1] # code as a string! We can search for `a` and `c` to make a list of variables to feed to Enzyme

# ╔═╡ 0ff88c23-8241-4095-9503-ad82700cbf67
print(funcs[1])

# ╔═╡ ffd2cdd0-5c13-42f3-b8dd-da05247eab16
md"""
#### Neat! 
We have a list of variables before `f1()` is run.

This is the part that's the most shady. Because we'll be replacing `f1()` with a version with the correct arguments the final version won't be playing around with the global scope, which would **definitely** torpedo performance if we weren't going to replace this. 

"""

# ╔═╡ b8c2dfaf-ea27-44ca-be9f-17c0fe98824b
md"""

This approach becomes feasible when there are *many* more blocks with mutation than just `f1()`.

---
**Note**

*Unfortunately*, at this point I have to completely abandon interactivity because `varinfo()` does not see `a` as a variable in ``Pluto``. From now on I'll show what you get if you were to run the code on your own machine. 


"""

# ╔═╡ 17ab01ff-f6a7-4c92-9cb4-5fae04c77ebe
md"""
---
If you'd like to see it here's the code that regexes through the simple example to produce the code from **Part 2**. Here's the output:
"""

# ╔═╡ 37565b52-2a63-452c-ad66-305c6048d5d0
mutfunc = "function f1!(a,c,val,i)\n\n\t\ta[i,i] = val\n\t\tc[i] = sin(val)\n\n return nothing\nend"

# ╔═╡ 77ec51ff-bfc9-4e74-9923-705d047f77a5
nonmutfunc = "function f1(a,c,val,i)\n\n\t\ta[i,i] = val\n\t\tc[i] = sin(val)\n return a,c\n end"

# ╔═╡ 4043cbe5-4cb2-4d7b-aa26-c41af6eff62e
zygadj = "Zygote.@adjoint function f1(a,c,val,i)\n\nf1!(a,c,val,i)\n\nfunction backf1(inmutvars)\n (∂z_a,∂z_c)=inmutvars \n∂z_a = ∂z_a isa Fill ? collect(∂z_a) : ∂z_a\n∂z_c = ∂z_c isa Fill ? collect(∂z_c) : ∂z_c\n∂z_a = ∂z_a≡nothing ? zero(a) : ∂z_a\n∂z_c = ∂z_c≡nothing ? zero(c) : ∂z_c\n∂z_val,∂_i=Enzyme.autodiff(\nf1!,Const,Duplicated(a,∂z_a),\nDuplicated(c,∂z_c),\nActive(val),\nActive(i))\nreturn ∂z_a,∂z_c,∂z_val,∂z_i\n end\nreturn (a,c),backf1\n end"

# ╔═╡ c06093a0-03c5-4b3b-a58e-2ee51abeab93
print(mutfunc)

# ╔═╡ 2a395fd2-59a1-41d4-8265-71f6da3e5de5
print(nonmutfunc)

# ╔═╡ 831ecbd0-1018-4ea7-aa7e-1b2d4cc285f6
print(zygadj)

# ╔═╡ 2df7fc7c-8614-4d61-9dcc-1f7b81edc126
md"""
 The script generates the pullback call for ``Enzyme`` that can then be copied and pasted in (and tested!).

It can also handle multiple mutating blocks that need to have pullbacks.

*Thank you for reading!*
"""


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CodeTracking = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
CodeTracking = "~1.0.9"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "6d4fa04343a7fc9f9cb9cff9558929f3d2752717"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.0.9"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
"""

# ╔═╡ Cell order:
# ╟─80901e1a-8088-4802-aaef-a6d3276a9004
# ╟─1f4a9328-8da1-4ffc-ab0d-97739ec546c1
# ╠═50346be3-9633-41cc-a929-b66865a5cb0f
# ╠═69a5f7a4-9be2-47e6-92af-a3b633497a08
# ╟─86653499-902e-40d1-9bb8-602e8aa21082
# ╠═6327deea-8e21-4bc6-9e25-16ddb056e5ce
# ╟─21064167-a0da-4e9e-9d9a-6244720fc6b5
# ╟─8f20995c-9813-45be-9342-faca9ad614b8
# ╟─8f14239b-85e0-496c-8fcb-ae5251ab1260
# ╟─09f76434-e162-46a0-986c-291e76aaf805
# ╟─8e34f06e-a247-4bcf-9903-2fb7672fc96e
# ╠═446832dd-f75e-41f3-8ce3-a38e58c87126
# ╟─b55fa59a-a81e-4b7b-aa26-6d88e6552812
# ╠═57ea9c88-ca05-4e31-a099-96a0072c15f4
# ╟─84b4c6dd-3b2d-4b67-aa67-383e0791a506
# ╠═81c0a936-6f11-4c0d-bc90-810bc52c4aca
# ╠═7670b446-a993-4d63-ba54-7e74690d1a52
# ╠═0ff88c23-8241-4095-9503-ad82700cbf67
# ╟─ffd2cdd0-5c13-42f3-b8dd-da05247eab16
# ╟─b8c2dfaf-ea27-44ca-be9f-17c0fe98824b
# ╟─17ab01ff-f6a7-4c92-9cb4-5fae04c77ebe
# ╟─37565b52-2a63-452c-ad66-305c6048d5d0
# ╟─77ec51ff-bfc9-4e74-9923-705d047f77a5
# ╟─4043cbe5-4cb2-4d7b-aa26-c41af6eff62e
# ╠═c06093a0-03c5-4b3b-a58e-2ee51abeab93
# ╠═2a395fd2-59a1-41d4-8265-71f6da3e5de5
# ╠═831ecbd0-1018-4ea7-aa7e-1b2d4cc285f6
# ╠═2df7fc7c-8614-4d61-9dcc-1f7b81edc126
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
