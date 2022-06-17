using Enzyme
using CodeTracking

"""
    Goal: 
        transform mutating block of code to Enzyme pullback to embed in Zygote
        to get around mutation issues in Zygote and BLAS issues in Enzyme

    Hacky solution: 
        use Julia's metaprogramming and regex abilities to build an Enzyme pullback 
        programatically and then embed this in Zygote (where one can use BLAS).

    How? There's several steps, but they should dramatically help rewriting of Enzyme code for Zygote.
        1. wrapping mutating block of code as a function and using CodeTracking package 
        to transform function to a string
        2. get variables defined until function is run to compare with variables used inside function (the mutating block)
        3. get variable input list in pullback by comparing the outputs of steps 1 and 2 with regex
        4. build enzyme pullback call
        5. populate the enzyme pullback function with inputs (from step 3)
        6. build mutating and nonmutating versions of mutating block (function from step 1)
        7. copy and paste step 5, 6 functions into a copy of the original code along with the function definition.
        They should work with Zygote.gradient and allow mutation without having to rewrite difficult
        fractions of the code (i.e. having to use buffers or map()).

    Rough patches: 
        1. Watch out when naming variables! Because it does regex then having a variable named
            `xp` might have issues if the `exp` function is used! Being more sophisticated with the regex
            would help with this.
        2. Need to manually define what values are Enzyme Consts (for me these are always integer values - could look for them that way)
        3. Constants in Enzyme are given a gradient of 0.0    
        
    Warning: metaprogramming is powerful but dangerous. Always check whether the output gives the same output as it did originally, especially when the 
    code was written by a geologist.
"""

# make some arbitrary matrix inputs
ndim = 3
a = zeros(ndim,ndim)
b = rand(ndim,ndim)
c = rand(ndim,1)

# store string-transformed functions (of mostly mutating sections) and variable lists
funcs = []
varsarr = []
varsmutarr = []
funcnames = []

for i in 1:ndim
	val = sum(b\c)    
	
	# get list of local and global variables defined up till f1() runs
	localf1 = string.(keys(Base.@locals)) # string array of local variables -> not needed if in global scope already
	globf1 = getindex.(getfield(varinfo().content[1],1),1) # string array of global variables
	varf1 = union(globf1,localf1)
	varf1mut = [var*r"\[" for var in varf1] # build list of parameters to check for mutation w. regex
	
    # wrap mutating block
    function f1()
		a[i,i] = val
		c[i] = sin(val)
	end
	f1()

	# Because of scoping issues this won't work in Pluto.jl, but it should work outside of pluto
	f1rawstr = @code_string f1() # make function a string to do regex on
    f1lines = split(f1rawstr,"\n")
	f1funcname = replace(f1lines[1],"function" => "")
    f1funcname = strip(replace(f1funcname,"()" => ""))
 	f1str = f1lines[2:end-1] # chop off function definition and "end" 	
	f1str = f1str.*"\n"
 	f1str = *(f1str...)

    # save everything once
	if varf1 ∉ varsarr
		push!(varsarr,varf1)
		push!(varsmutarr,varf1mut)
		push!(funcs,f1str)
        push!(funcnames,f1funcname)
	end 
end

# this works when there are a lot of mutating blocks like f1()
for (func,vars,varsmut,funcname) in zip(funcs,varsarr,varsmutarr,funcnames)

    """
        find non mutating and mutating variables
    """
    is_var = occursin.(vars,func)
    
    # make string list of all input variables
    invars = vars[is_var]
    invarlist = [var*"," for var in invars[1:end-1]]
    invarlist = push!(invarlist,invars[end])
    invarstr = *(invarlist...)

    # constant variables (not constants - usually ints)
    constinvars = [] # can't find a better way than listing consts
    if !isempty(constinvars)
        constinvarstr = *(constinvars...)
        indconst = findall(x->x in constinvars,invars)  
    else 
        constinvarstr = ""
        indconst = []
    end
    
    # mutating variables 
    # first need to find them 
    # next, need to make sure actually mutation happening and not simply indexing
    is_varmut = zeros(Bool,length(varsmut))
    for (ind, varmut) in enumerate(varsmut)
        lines = strip.(split(func,"\n"))
        for line in lines
            if startswith(line,varmut)
                is_varmut[ind] = true
            end    
        end
    end

    # get mutating variables
    mutvars = vars[is_varmut]
    mutlist = []
    mutlist = [var*"," for var in mutvars[1:end-1]]
    mutlist = push!(mutlist,mutvars[end])
    mutvarstr = *(mutlist...)
    print("Found mutating variables $(mutvarstr) in function $(funcname)! \n")

    # get nonmutating variables
    # these are needed to preallocate gradients for Enzyme
    nonmutvars = vars[(.!is_varmut .& is_var)]
    nonmutvars = [nonmutinvar for nonmutinvar in nonmutvars if !(nonmutinvar in constinvars)]
    nonmutlist = ["∂z_"*var*"," for var in nonmutvars[1:end-1]]
    nonmutlist = push!(nonmutlist,"∂_"*nonmutvars[end])

    """ 
        once all variables are processed
        build Enzyme pullback
    """
    mainfunc = funcname*"("*invarstr*")\n"
    mainfuncmut = funcname*"!("*invarstr*")\n"
    backfunc = "function back"*funcname*"(inmutvars)\n "
    mutgrads =  ["∂z_"*var for var in mutvars]
    mutgradlist = [var*"," for var in mutgrads[1:end-1]]
    mutgradlist = push!(mutgradlist,mutgrads[end])
    backfunc = backfunc*"($(mutgradlist...))=inmutvars \n"

    # handle edge case with nothings and with Fill type - both come from Zygote can cause errors in Enzyme 
    checkFill = [mutgrad*" = "*mutgrad*" isa Fill ? collect("*mutgrad*") : "*mutgrad*"\n" for mutgrad in mutgrads]
    checkNothing = [mutgrad*" = "*mutgrad*"≡nothing ? zero("*mutparam*") : "*mutgrad*"\n" for (mutgrad,mutparam) in zip(mutgrads,mutvars)]
    enzymecall = "Enzyme.autodiff(\n"*funcname*"!,"
    autodiff_inlist = ["Const,"]

    # build input into Enzyme call
    # based on type of call (duplicated -> mutating; active -> "tracked;" const -> "not tracking gradient")
    for (index,param) in enumerate(invars)
        # beginning of list -> need to end in ,
        if index!=length(invars)
            if occursin(param,mutvarstr) 
                newparam = "Duplicated("*param*","*mutgrads[mutvars.==param][1]*"),\n"
                push!(autodiff_inlist,newparam)
            elseif occursin(param,constinvarstr)
                newparam = "Const("*param*"),\n"
                push!(autodiff_inlist,newparam)            
            elseif occursin(param,invarstr)
                newparam = "Active("*param*"),\n"
                push!(autodiff_inlist,newparam)
            end
        # case if at end of list -> need to end in )
        elseif index==length(invars)
            if occursin(param,mutvarstr) && index==length(invars)
                newparam = "Duplicated("*param*","*mutgrads[mutvars.==param][1]*"))\n"
                push!(autodiff_inlist,newparam)
            elseif occursin(param,constinvarstr)
                newparam = "Const("*param*"))\n"
                push!(autodiff_inlist,newparam)   
            elseif occursin(param,invarstr)
                newparam = "Active("*param*"))\n"
                push!(autodiff_inlist,newparam)
            end
        end
    end

    # assemble up to output
    enzymecall = enzymecall*"$(autodiff_inlist...)"
    enzymecall = *(nonmutlist...)*"="*enzymecall

    # output of pullback; assign 0 if constant and check for 0 if active 
    outback = vars[(is_var)]
    for ind in indconst
        outback[ind] = "0.0"
    end

    for (ind,var) in enumerate(outback) 
        if var!="0.0"
            outback[ind] = "∂z_"*var
        end
        if ind!=length(outback)
            outback[ind] = outback[ind]*","
        end
    end

    enzymecall = enzymecall*"return "*(outback...)*"\n end\n"
    backcall = "\n"*backfunc*(checkFill...)*(checkNothing...)*enzymecall

    # pullback and input function get returned
    global zygadj = "Zygote.@adjoint function "*mainfunc*"\n"*mainfuncmut*backcall*"return ("*(mutlist...)*"),back"*funcname*"\n end"
    global mutfunc = "function "*mainfuncmut*"\n"*func*"\n return nothing\nend"
    global nonmutfunc = "function "*mainfunc*"\n"*func*" return "*(mutlist...)*"\n end"
    print("\n Mutating function to replace $(funcname) is: \n\n"*mutfunc)
    print("\n\nNon-mutating version of $(funcname) is: \n\n"*nonmutfunc)
    print("\n\nZygote.@adjoint is: \n\n"*zygadj)
end