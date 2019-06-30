
BeginPackage["SymbolicC`SymbolicCUDA`"]


SymbolicCUDAKernelCall::usage = "SymbolicCUDAKernelCall[ fname, {gridDim, blockDim, sharedMem}, args] is a symbolic representation of a call to a cuda kernel function."


Begin["`Private`"]

Needs["SymbolicC`"]

CPrecedence[_SymbolicCUDAKernelLaunch] = CPrecedence[CCall] 

SymbolicC`Private`IsCExpression[ _SymbolicCUDAKernelCall] := True


GenerateCode[SymbolicCUDAKernelCall[funName_, {gridDim_, blockDim_, sharedMem_:None}, args_], opts:OptionsPattern[]] :=
	GenerateCode[
		CCall[
			StringJoin[{
				GenerateCode[funName, opts],
				 "<<<",
				 Riffle[
				 	GenerateCode[#, opts]& /@ {gridDim, blockDim, If[sharedMem === None, Nothing, sharedMem]},
				 	","
				 ],
				 ">>>"
			}],
			args
		],
		opts
	]
		
    
End[]

EndPackage[]