
addprocs(CPU_CORES)


@everywhere const n=20
@everywhere const sdJ=1/sqrt(n)
@everywhere const meanJ=0


trials=600

@everywhere function bitarray(int::Int)  
    2*digits(int,2,n)-1
end

@everywhere function energy(index, J)
    c=bitarray(index)
    -sum(J.*reshape(kron(c,c),(n,n)))
end


@everywhere function distance(a::Int,b::Int)     
    min(sum(digits(a$b,2,n)),sum(digits((2^(n+1)-1-a)$b,2,n)))
end

@everywhere function energydifferance(energies)   
    lowenergy=findmin(energies)
    energies[lowenergy[2]]=NaN
    secondlow=findmin(energies)
    [lowenergy[1]-secondlow[1],distance(lowenergy[2]-1,secondlow[2]-1)]
end
@everywhere function energyinstance(J)
    energydifferance([energy(i,J) for i=0:2^(n-1)-1])
end

gapdistance=@parallel (vcat) for i=1:trials
    transpose(energyinstance(randn(n,n)*sdJ+meanJ))
end


rmprocs(workers())

using PyPlot

PyPlot.plt.hist(gapdistance[:,2],ceil(n/2))
title("histogram of required flips ")
ylabel("number of states")
xlabel("flips")
show()

PyPlot.plt.hist(gapdistance[:,1],40)
title("histogram of the energy gap ")
ylabel("number of states")
xlabel("energy gap")
show()

scatter(gapdistance[:,2],gapdistance[:,1],color="black")
title("energy gap vs required flips")
xlabel("required flips")
ylabel("energy gap")
show()
