# hybrid AD
Here are some Pluto.jl workbooks for a first cut at a temporary workaround for getting fast autodiff with mutation and BLAS operations by combining
``Enzyme`` and ``Zygote`` via ``Enzyme`` pullbacks.

---

To **run** these notebooks in **Binder** use [Pluto on Binder!](https://pluto-on-binder.glitch.me/)

You can copy the *first* workbook link at 
``` 
https://github.com/ryanstoner1/hybridADpluto/blob/main/enzymezygote1.jl 
```
and so on.

Startup will probably take a couple of minutes. 

This was mostly inspired by a Discourse post made by Jordi Bolibar [here](https://discourse.julialang.org/t/open-discussion-on-the-state-of-differentiable-physics-in-julia/72900) and follow-up blog [here](http://www.stochasticlifestyle.com/engineering-trade-offs-in-automatic-differentiation-from-tensorflow-and-pytorch-to-jax-and-julia/) by Chris Rackaukas in addition to my own personal travails with `Zygote`.  
