(* 
Perform index propagation (see Fink, Knobe & Sarkar, SAS 2000)
This analysis computes for each Array SSA variable A, the set of value numbers V(k) such that location A[k] is "available" at def A, and thus at all uses of A

We formulate this as a data flow problem as described in the paper.

https://www.jikesrvm.org/JavaDoc/org/jikesrvm/compilers/opt/ssa/IndexPropagation.html
 *)