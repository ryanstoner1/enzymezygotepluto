
using Zygote 
using Enzyme
using FillArrays


function f1!(a,c,val,i)
	a[i,i] = val
	c[i] = sin(val)
    return nothing
end

function f1(a,c,val,i)
	f1!(a,c,val,i)
 	return a,c
end

Zygote.@adjoint function f1(a,c,val,i)
	f1!(a,c,val,i)
	function backf1(inmutvars)
		(∂z_a,∂z_c)=inmutvars 
		∂z_a = ∂z_a isa Fill ? collect(∂z_a) : ∂z_a # check for fills that will make Enzyme unhappy
		∂z_c = ∂z_c isa Fill ? collect(∂z_c) : ∂z_c		
        ∂z_a = ∂z_a≡nothing ? zero(a) : ∂z_a # check for nothings that will make Enzyme unhappy
		∂z_c = ∂z_c≡nothing ? zero(c) : ∂z_c

		(∂z_val,)=Enzyme.autodiff(
			f1!,Const,Duplicated(a,∂z_a),
			Duplicated(c,∂z_c),
			Active(val),
			i)

		return ∂z_a,∂z_c,∂z_val,0.0
	end
	return (a,c),backf1
 end

n = 3
b = rand(n,n)

function hybrid(b,n)
	a = zeros(n,n) # preallocate
	c = rand(n,1)
	for i in 1:n
		val = sum(b\c)
		a,c = f1(a,c,val,i)
	end
	return sum(a)
end

gradout = Zygote.gradient(hybrid,b,n)

