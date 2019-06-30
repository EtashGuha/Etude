(* ::Package:: *)

(* Mathematica package *)

BeginPackage["GPUTools`SymbolicGPU`", {"SymbolicC`"}]

GPUKernelFunction::usage = "GPUKernelFunction  "

GPUKernelThreadIndex::usage = "GPUKernelThreadIndex  "

GPUKernelBlockIndex::usage = "GPUKernelBlockIndex  "

GPUKernelBlockDimension::usage = "GPUKernelBlockDimension  "

GPUCalculateKernelIndex::usage = "GPUCalculateKernelIndex  "

GPUDeclareIndexBlock::usage = "GPUDeclareIndexBlock  "

GPUKernelIndex::usage = "GPUKernelIndex  "

GPUExpression::usage = "GPUExpression  "

GPUKernel::usage = "GPUKernel  "

KernelBlockIndex::usage = "KernelBlockIndex  "

KernelThreadIndex::usage = "KernelThreadIndex  "

KernelBlockDimension::usage = "KernelBlockDimension  "

CalculateKernelIndex::usage = "CalculateKernelIndex  "

DeclareIndexBlock::usage = "DeclareIndexBlock  "

Begin["Private`"]


$CUDAAPI = "CUDA"
$OpenCLAPI = "OpenCL"


$SymbolicGPUOptions = Sort@{
	"APITarget" -> "CUDA",
	"TargetPrecision" -> "Double"
}

GetAndCheckOption[head_, opts_List, n:"TargetPrecision"] :=
	Module[{opt = OptionValue[head, opts, n]},
		If[MemberQ[{"Single", "Double"}, opt],
			opt === "Single",
			Message[head::invtrgtprs, opt];
			$Failed
		]
	]


GetAndCheckOption[head_, opts_List, n:"APITarget"] :=
	Module[{opt = OptionValue[head, opts, n]},
		If[MemberQ[{$CUDAAPI, $OpenCLAPI}, opt],
			opt,
			Message[head::invtapi, opt];
			$Failed
		]
	]

CPrecedence[GPUExpression] = CPrecedence[CExpression]
CPrecedence[_GPUExpression] = CPrecedence[GPUExpression] 

CPrecedence[iGPUExpression] = CPrecedence[CExpression]
CPrecedence[_iGPUExpression] = CPrecedence[iGPUExpression] 
SymbolicC`Private`IsCExpression[ _iGPUExpression] := True

SetAttributes[GPUExpression, HoldAll]
SetAttributes[iGPUExpression, HoldAll]

GenerateCode[GPUExpression[x___], opts:OptionsPattern[]] := GenerateCode[Quiet[iGPUExpression[x, opts]]]

Options[GPUExpression] := $SymbolicGPUOptions
GPUExpression[x___, opts:OptionsPattern[]] := Quiet[iGPUExpression[x, opts]]

Options[iGPUExpression] := $SymbolicGPUOptions

iGPUExpression[Set[x_, If[cond_, true_, false_]], opts:OptionsPattern[]] := CAssign[iGPUExpression[x, opts], CConditional[iGPUExpression[cond, opts], iGPUExpression[true, opts], iGPUExpression[false, opts]]]
iGPUExpression[Set[x_List, y_List], opts:OptionsPattern[]] /; Length[x] === Length[y] := MapThread[CAssign[iGPUExpression[#1, opts], iGPUExpression[#2, opts]]&, {x, y}]
iGPUExpression[Set[x_, y_], opts:OptionsPattern[]] := CAssign[iGPUExpression[x, opts], iGPUExpression[y, opts]]

iGPUExpression[CompoundExpression[stmts__], opts:OptionsPattern[]] := Quiet[iGPUExpression[{stmts}, opts]]
	
iGPUExpression[iGPUExpression[x___, opts:OptionsPattern[]], opts:OptionsPattern[]] := iGPUExpression[x, opts]

iGPUExpression[OddQ[x_], opts:OptionsPattern[]] := COperator[BitAnd, {x, 1}]
iGPUExpression[EvenQ[x_], opts:OptionsPattern[]] := COperator[Equal, {COperator[Mod, {x, 2}], 0}]
iGPUExpression[TrueQ[x_], opts:OptionsPattern[]] := COperator[Equal, {x, True}]
iGPUExpression[Not[TrueQ[x_]], opts:OptionsPattern[]] := COperator[Equal, {x, False}]

iGPUExpression[(h:(Minus | BitNot | Not | Decrement | Increment | PreDecrement | PreIncrement))[x_], opts:OptionsPattern[]] := COperator[h, iGPUExpression[x, opts]]


iGPUExpression[(h:(Mod | Divide | Subtract | BitShiftRight | BitShiftLeft))[x_, y_], opts:OptionsPattern[]] := COperator[h, {iGPUExpression[x, opts], iGPUExpression[y, opts]}]

iGPUExpression[Times[-1, x_], opts:OptionsPattern[]] := "-" <> GenerateCode[iGPUExpression[x, opts]]
iGPUExpression[Times[x__], opts:OptionsPattern[]] := COperator[Times, iGPUExpression[#, opts]& /@ {x}]
iGPUExpression[Plus[x_, Times[-1, y__]], opts:OptionsPattern[]] := COperator[Subtract, iGPUExpression[#, opts]& /@ Flatten[{x, y}]]
iGPUExpression[Plus[x__], opts:OptionsPattern[]] /; Select[{x}, Negative, 1] =!= {} := 
	Module[{negVals = Select[{x}, Negative], posVals, posExpr},
		posVals = Complement[{x}, negVals];
		posExpr = Switch[Length[posVals],
			0,
				{},
			1,
				First[posVals],
			_,
				COperator[Plus, iGPUExpression[#, opts]& /@ posVals]
		];
		COperator[Subtract, Join[Flatten[{posExpr}], iGPUExpression[#, opts]& /@ -1*negVals]]
	]
iGPUExpression[Plus[x__], opts:OptionsPattern[]] /; Select[{x}, Negative, 1] === {} := COperator[Plus, iGPUExpression[#, opts]& /@ {x}]

iGPUExpression[(h:(Less | Greater | GreaterEqual | LessEqual))[x__], opts:OptionsPattern[]] := COperator[h, iGPUExpression[#, opts]& /@ {x}]

iGPUExpression[UnsameQ[x__], opts:OptionsPattern[]] := COperator[Unequal, iGPUExpression[#, opts] /@ {x}]
iGPUExpression[SameQ[x__], opts:OptionsPattern[]] := COperator[Equal, iGPUExpression[#, opts] /@ {x}]

iGPUExpression[(h:(Equal | Unequal | BitAnd | BitXor | And | Or))[x__], opts:OptionsPattern[]] := COperator[h, iGPUExpression[#, opts]& /@ {x}]

iGPUExpression[(h:(AddTo | SubtractFrom | TimesBy | DivideBy))[x_, y_], opts:OptionsPattern[]] := CAssign[h, iGPUExpression[x, opts], iGPUExpression[y, opts]]

iGPUExpression[(h:(ArcCos | ArcSin | Ceiling | Cos | Cosh | Exp | Abs | Floor | Sin | Sinh | Sqrt | Tan | Tanh | Log))[x_], opts:OptionsPattern[]] :=
	CStandardMathOperator[h, {iGPUExpression[x, opts]}]
iGPUExpression[ArcTan[x_, y_], opts:OptionsPattern[]] := CStandardMathOperator[ArcTan, {iGPUExpression[x, opts], iGPUExpression[y, opts]}]

iGPUExpression[Csc[x_], opts:OptionsPattern[]] := COperator[Divide, {1.0, CStandardMathOperator[Sin, {iGPUExpression[x, opts]}]}]
iGPUExpression[Cot[x_], opts:OptionsPattern[]] := COperator[Divide, {1.0, CStandardMathOperator[Tan, {iGPUExpression[x, opts]}]}]
iGPUExpression[Sec[x_], opts:OptionsPattern[]] := COperator[Divide, {1.0, CStandardMathOperator[Cos, {iGPUExpression[x, opts]}]}]

iGPUExpression[Csch[x_], opts:OptionsPattern[]] := COperator[Divide, {1.0, CStandardMathOperator[Sinh, {iGPUExpression[x, opts]}]}]
iGPUExpression[Coth[x_], opts:OptionsPattern[]] := COperator[Divide, {1.0, CStandardMathOperator[Tanh, {iGPUExpression[x, opts]}]}]
iGPUExpression[Sech[x_], opts:OptionsPattern[]] := COperator[Divide, {1.0, CStandardMathOperator[Cosh, {iGPUExpression[x, opts]}]}]

iGPUExpression[ArcCsc[x_], opts:OptionsPattern[]] := CStandardMathOperator[ArcSin, {COperator[Divide, {1, iGPUExpression[x, opts]}]}]
iGPUExpression[ArcCot[x_], opts:OptionsPattern[]] := CStandardMathOperator[ArcTan, {COperator[Divide, {1, iGPUExpression[x, opts]}]}]
iGPUExpression[ArcSec[x_], opts:OptionsPattern[]] := CStandardMathOperator[ArcCos, {COperator[Divide, {1, iGPUExpression[x, opts]}]}]

iGPUExpression[ArcCosh[x_], opts:OptionsPattern[]] :=
	CStandardMathOperator[Log, {COperator[Plus, {iGPUExpression[x, opts], COperator[Sqrt, {COperator[Subtract, {CStandardMathOperator[Power, {iGPUExpression[x, opts], 2.0}]}]}]}]}]
iGPUExpression[ArcSinh[x_], opts:OptionsPattern[]] :=
	CStandardMathOperator[Log, {COperator[Plus, {iGPUExpression[x, opts], COperator[Sqrt, {COperator[Plus, {CStandardMathOperator[Power, {iGPUExpression[x, opts], 2.0}]}]}]}]}]
iGPUExpression[ArcTanh[Divide[x_, y_]], opts:OptionsPattern[]] :=
	With[{xc = iGPUExpression[x, opts], yc = iGPUExpression[y, opts]},
		CStandardMathOperator[Log, {COperator[Divide, {COperator[Divide, {COperator[Plus, {yc, xc}], COperator[Subtract, {yc, xc}]}], 2.0}]}]
	]
iGPUExpression[ArcTanh[x_], opts:OptionsPattern[]] :=
	With[{xc = iGPUExpression[x, opts]},
		CStandardMathOperator[Log, {COperator[Divide, {COperator[Divide, {COperator[Plus, {1.0, xc}], COperator[Subtract, {1.0, xc}]}], 2.0}]}]
	]

iGPUExpression[Power[x_, Rational[1, 2]], opts:OptionsPattern[]] := CStandardMathOperator[Sqrt, {iGPUExpression[x, opts]}]
iGPUExpression[Power[x_, Rational[-1, 2]], opts:OptionsPattern[]] := COperator[Divide, {1.0, CStandardMathOperator[Sqrt, {iGPUExpression[x, opts]}]}]
iGPUExpression[Power[x_, r:Rational[_, _]], opts:OptionsPattern[]] := CStandardMathOperator[Power, {iGPUExpression[x, opts], iGPUExpression[r, opts]}]
iGPUExpression[Power[x_, 2], opts:OptionsPattern[]] := COperator[Times, {iGPUExpression[x, opts], iGPUExpression[x, opts]}]
iGPUExpression[Power[x_, y_], opts:OptionsPattern[]] := CStandardMathOperator[Power, {iGPUExpression[x, opts], iGPUExpression[y, opts]}]

iGPUExpression[Part[x_, idx_], opts:OptionsPattern[]] := CArray[iGPUExpression[x, opts], iGPUExpression[idx, opts]]
iGPUExpression[Return[], opts:OptionsPattern[]] := CReturn[]
iGPUExpression[Return[ret_], opts:OptionsPattern[]] := CReturn[iGPUExpression[ret, opts]]
iGPUExpression[CAssign[x_, If[y___]], opts:OptionsPattern[]] := CAssign[x, CConditional[iGPUExpression[#, opts]& /@ {y}]]
iGPUExpression[If[cond_, trueStmt_], opts:OptionsPattern[]] := CIf[iGPUExpression[cond, opts], iGPUExpression[trueStmt, opts]]
iGPUExpression[If[cond_, trueStmt_, falseStmt_], opts:OptionsPattern[]] := CIf[iGPUExpression[cond, opts], iGPUExpression[trueStmt, opts], iGPUExpression[falseStmt, opts]]

(*
TODO: Think about, since this is not equivalent to M Syntax
GenerateCode[GPUExpression[Do[body_, cond_]], opts:OptionsPattern[]] := GenerateCode[CDo[GPUExpression[body], GPUExpression[cond]]
*)
  
iGPUExpression[For[init_, cond_, inc_, body__], opts:OptionsPattern[]] := CFor[iGPUExpression[init, opts], iGPUExpression[cond, opts], iGPUExpression[inc, opts], iGPUExpression[body, opts]]
iGPUExpression[Switch[args__], opts:OptionsPattern[]] := Apply[CSwitch, ReplaceAll[iGPUExpression[{args}, opts], {Verbatim[CExpression[_]] -> CDefault[]}]]

iGPUExpression[Which[args0__], opts:OptionsPattern[]] :=
	Module[{expr, TmpIf, args=iGPUExpression[{args0}, opts]},
		TmpIf[{}] := {};
		TmpIf[CIf[x__], u:{}] := CIf[x];
		TmpIf[CIf[x_, y_], {CIf[cond:(True | "True" | False | "False"), z_]}] := CIf[x, y, z];
		TmpIf[CIf[x_, y_], {CIf[a___]}] := CIf[x, y, CIf[a]];
		TmpIf[a_CIf, b_] := TmpIf[a, {b}];
		TmpIf[x_List] := TmpIf[First[x], TmpIf[Rest[x]]];
		expr = Join[MapThread[CIf, Transpose@Partition[args, 2]]];
		TmpIf[expr]
	]

iGPUExpression[Module[vars_List, body_], opts:OptionsPattern[]] := {iDeclareVariables[{iGPUExpression[vars, opts]}, body, opts] ~Join~ {iGPUExpression[body, opts]}}
iGPUExpression[Block[vars_List, body_], opts:OptionsPattern[]] := iDeclareVariables[{iGPUExpression[vars, opts]}, body, opts] ~Join~ {iGPUExpression[body, opts]}
iGPUExpression[With[vars_List, body_], opts:OptionsPattern[]] := iDeclareVariables[{iGPUExpression[vars, opts]}, body, opts] ~Join~ {iGPUExpression[body, opts]}

SetAttributes[iDeclareVariables, HoldAll]
iDeclareVariables[{}, x___, opts:OptionsPattern[]] := {}
iDeclareVariables[vard_List, body0_, opts:OptionsPattern[]] :=
	Module[{body = Flatten[List[iGPUExpression[body0, opts]]], usage, lhs, varname},
		Flatten[Map[
			Function[{var},
				varname = If[MatchQ[var, CExpression[x_String]],
					First[var],
					var
				];
				usage = Select[body, MatchQ[#, CAssign[varname, _]]&, 1];
				If[usage === {},
					Switch[var,
						_CAssign,
							CDeclare[GPUType[var, opts], var],
						_,
							{}
					],
					lhs = First[usage] //. CAssign[ToString[var] | var, x___] -> x;
					CDeclare[GPUType[lhs, opts], var]
				]
			], Flatten[vard]
		]]
	]

Options[GPUType] := $SymbolicGPUOptions
GPUType[Integer, opts:OptionsPattern[]] := "mint"
GPUType[Real, opts:OptionsPattern[]] :=
    If[GetAndCheckOption[GPUType, {opts}, "TargetPrecision"],
        "float",
        "double"
    ]
GPUType[Complex, opts:OptionsPattern[]] := "complex"
GPUType[True | "True", opts:OptionsPattern[]] := "mbool"
GPUType[False | "False", opts:OptionsPattern[]] := "mbool"

GPUType[x_String, opts:OptionsPattern[]] /; (StringMatchQ[x, (DigitCharacter | ".").. ~~ "f"] && TrueQ[Quiet@NumericQ[ToExpression[StringTrim[x, "f"]]]]) := GPUType[Real, opts]

GPUType[_Integer, opts:OptionsPattern[]] := GPUType[Integer, opts]
GPUType[_Real, opts:OptionsPattern[]] := GPUType[Real, opts]
GPUType[_Complex, opts:OptionsPattern[]] := GPUType[Complex, opts]

GPUType[CAssign[_, t___], opts:OptionsPattern[]] := GPUType[{t}, opts]

GPUType[x_List, opts:OptionsPattern[]] :=
	With[{types = GPUType /@ x},
		Which[
			MemberQ[types, GPUType[Complex, opts]],
				GPUType[Complex, opts],
			MemberQ[types, GPUType[Real, opts]],
				GPUType[Real, opts],
			MemberQ[types, GPUType[Integer, opts]],
				GPUType[Integer, opts],
			MemberQ[types, GPUType[True, opts]],
				GPUType[True, opts],
			MemberQ[types, GPUType[False, opts]],
				GPUType[False, opts],
			True,
				GPUType[Integer, opts]
				(* "UnknownType" *)
		]
	]

GPUType[CExpression[x_], opts:OptionsPattern[]] := GPUType[x, opts]

GPUType[COperator[_, args__], opts:OptionsPattern[]] := GPUType[args, opts] 
  
iGPUExpression[Apply[f_, args__], opts:OptionsPattern[]] := iGPUExpression[f[args], opts]

iGPUExpression[arg_Symbol, opts:OptionsPattern[]] :=
	ToString[arg]
iGPUExpression[arg_Integer, opts:OptionsPattern[]] :=
	CExpression[arg]
iGPUExpression[{}, opts:OptionsPattern[]] := {}
iGPUExpression[Null, opts:OptionsPattern[]] := {}
iGPUExpression[args:{stmts__}, opts:OptionsPattern[]] :=
	List[
		ReleaseHold[
			Map[
				Function[stmt,
					iGPUExpression[stmt, opts]
					, {HoldAll}
				], Hold[stmts]
			]
		]		
  	]
	

SetAttributes[GPUKernel, HoldAll]
Options[GPUKernel] := $SymbolicGPUOptions
GPUKernel[vars_List, body_, opts:OptionsPattern[]] :=
	GPUKernel[None, vars, body, opts]
GPUKernel[name:(_String | None), vars_List, body_, opts:OptionsPattern[]] :=
	{
		If[GetAndCheckOption[GPUKernel, {opts}, "APITarget"] === $OpenCLAPI && GetAndCheckOption[GPUKernel, {opts}, "TargetPrecision"] === False,
			CPragma["OPENCL EXTENSION cl_khr_fp64: enable"],
			{}
		],
		CFunction[
			If[GetAndCheckOption[GPUKernel, {opts}, "APITarget"] === $CUDAAPI,
				{"__global__", "void"},
				{"__kernel", "void"}
			],
			If[name === None, "Unique"["gpuf"], name],
			Map[GPUKernelParamType[#, body, opts]&, vars],
			CBlock[{iGPUExpression[body, opts]}]
		]
	}

SetAttributes[GPUKernelParamType, HoldAll]
Options[GPUKernelParamType] := $SymbolicGPUOptions
GPUKernelParamType[varName_Symbol, body_, opts:OptionsPattern[]] := Flatten[iDeclareVariables[{iGPUExpression[varName, opts]}, body] /. CDeclare[x_, y_] -> {x, y}]
GPUKernelParamType[{varName_, type_}, body_, opts:OptionsPattern[]] := 
	{If[ListQ[type], type, GPUType[First@type, opts]], varName}
GPUKernelParamType[{varName_, type0_, rank_}, body_, opts:OptionsPattern[]] :=
	With[{type = If[ListQ[type0], type0, GPUType[First@type0, opts]]},
		{
			If[GetAndCheckOption[GPUKernelParamType, {opts}, "APITarget"] === $CUDAAPI,
				CPointerType[type],
				CPointerType[{"__global", type}]
			]
			, varName
		}
	]

GPUType[CMember["threadIdx", _], opts:OptionsPattern[]] := GPUType[Integer, opts]
GPUType[CMember["blockIdx", _], opts:OptionsPattern[]] := GPUType[Integer, opts]
GPUType[CMember["blockDim", _], opts:OptionsPattern[]] := GPUType[Integer, opts]

GPUType[CCall["get_global_id", _], opts:OptionsPattern[]] := GPUType[Integer, opts]
GPUType[CCall["get_local_id", _], opts:OptionsPattern[]] := GPUType[Integer, opts]
GPUType[CCall["get_global_size", _], opts:OptionsPattern[]] := GPUType[Integer, opts]
GPUType[CCall["get_local_size", _], opts:OptionsPattern[]] := GPUType[Integer, opts]

GPUKernelFunction[api:$CUDAAPI, funName_, args_List] := 
	CFunction[{"__global__", "void"}, funName, Select[args, # =!= {} &]]
GPUKernelFunction[api:$CUDAAPI, funName_, args_List, body_] :=
	CFunction[{"__global__", "void"}, funName, Select[args, # =!= {} &], body]
GPUKernelFunction[api:$OpenCLAPI, funName_, args_List] := 
	CFunction[{"__kernel", "void"}, funName, Select[args, # =!= {} &]]
GPUKernelFunction[api:$OpenCLAPI, funName_, args_List, body_] :=
	CFunction[{"__kernel", "void"}, funName, Select[args, # =!= {} &], body]

GPUKernelThreadIndex[api:$CUDAAPI, idx_: 1] /; MemberQ[Range[3], idx] :=
    CMember["threadIdx", {"x", "y", "z"}[[idx]]]
GPUKernelBlockIndex[api:$CUDAAPI, idx_: 1] /; MemberQ[Range[3], idx] :=
    CMember["blockIdx", {"x", "y", "z"}[[idx]]]
GPUKernelBlockDimension[api:$CUDAAPI, idx_: 1] /; MemberQ[Range[3], idx] :=
    CMember["blockDim", {"x", "y", "z"}[[idx]]]

GPUKernelThreadIndex[api:$OpenCLAPI, idx_: 1] /; MemberQ[Range[3], idx] :=
    CCall["get_local_id", {idx-1}]
GPUKernelBlockIndex[api:$OpenCLAPI, idx_: 1] /; MemberQ[Range[3], idx] :=
    CCall["get_group_id", {idx-1}]
GPUKernelBlockDimension[api:$OpenCLAPI, idx_: 1] /; MemberQ[Range[3], idx] :=
    CCall["get_local_size", {idx-1}]    


ValidDimensionQ[d:(1 | 2 | 3 | "x" | "y" | "z" | "X" | "Y" | "Z")] := True
ValidDimensionQ[___] := False

IntegerDimension[d_] :=
	Switch[d,
		1 | "x" | "X",
			1,
		2 | "y" | "Y",
			2,
		3 | "z" | "Z",
			3
	]

iGPUExpression[KernelBlockIndex[d_?ValidDimensionQ], opts:OptionsPattern[]] :=
	GPUKernelBlockIndex[GetAndCheckOption[iGPUExpression, {opts}, "APITarget"], IntegerDimension[d]]
	
iGPUExpression[KernelThreadIndex[d_?ValidDimensionQ], opts:OptionsPattern[]] :=
	GPUKernelThreadIndex[GetAndCheckOption[iGPUExpression, {opts}, "APITarget"], IntegerDimension[d]]
	
iGPUExpression[KernelBlockDimension[d_?ValidDimensionQ], opts:OptionsPattern[]] :=
	GPUKernelBlockDimension[GetAndCheckOption[iGPUExpression, {opts}, "APITarget"], IntegerDimension[d]]

GPUKernelIndex[api:$OpenCLAPI, dim_:1] :=
	CCall["get_global_id", {dim-1}]    
GPUKernelIndex[api:$CUDAAPI, dim_: 1] :=
    COperator[Plus, {
    	GPUKernelThreadIndex[api, dim],
      	COperator[Times, {
        	GPUKernelBlockIndex[api, dim],
        	GPUKernelBlockDimension[api, dim]
		}]
	}]
GPUCalculateKernelIndex[api:($CUDAAPI | $OpenCLAPI), dim_, pitch_: 1, depth_: 1] /; MemberQ[Range[3], dim]:=
	Switch[dim,
    	1,
    		GPUKernelIndex[api, dim],
    	2,
    		COperator[Plus, {
    			GPUKernelIndex[api, 1],
       			COperator[Times, {
         			pitch,
         			GPUKernelIndex[api, 2]
         		}]
       		}],
       	3,
       		COperator[Plus, {
    			GPUKernelIndex[api, 1],
       			COperator[Times, {
         			pitch,
					COperator[Plus, {
						GPUKernelIndex[api, 2],         			
         				COperator[Times, {
         					depth,
         					GPUKernelIndex[api, 3]
         				}]
					}]
         		}]
       		}]
     ]

iGPUExpression[CalculateKernelIndex[d_?ValidDimensionQ, pitch_: 1, depth_: 1], opts:OptionsPattern[]] :=
	GPUCalculateKernelIndex[GetAndCheckOption[iGPUExpression, {opts}, "APITarget"], IntegerDimension[d], pitch, depth]

GPUDeclareIndexBlock[api:($CUDAAPI | $OpenCLAPI), 1, ___] :=
	{
		CDeclare["mint", CAssign["index", GPUKernelIndex[api, 1]]]
	}
GPUDeclareIndexBlock[api:($CUDAAPI | $OpenCLAPI), 2, pitch_:"width", ___] :=
	{
		CDeclare["mint", CAssign["xIndex", GPUKernelIndex[api, 1]]],
		CDeclare["mint", CAssign["yIndex", GPUKernelIndex[api, 2]]],
		CDeclare["mint", 
			CAssign["index", 
				COperator[Plus, {
					"xIndex",
					COperator[Times, {pitch, "yIndex"}]
				}]
			]
		]
	}
GPUDeclareIndexBlock[api:($CUDAAPI | $OpenCLAPI), 3, pitch_:"width", depth_:"height"] :=
	{
		CDeclare["mint", CAssign["xIndex", GPUKernelIndex[api, 1]]],
		CDeclare["mint", CAssign["yIndex", GPUKernelIndex[api, 2]]],
		CDeclare["mint", CAssign["zIndex", GPUKernelIndex[api, 3]]],
		CDeclare["mint", 
			CAssign["index", 
				COperator[Plus, {
					"xIndex",
					COperator[Times, {
						pitch,
						COperator[Plus, {
							"yIndex",
							COperator[Times, {depth, "zIndex"}]														
						}]
					}]
				}]
			]
		]
	}

iGPUExpression[DeclareIndexBlock[d_?ValidDimensionQ, pitch_:"width", depth_:"height"], opts:OptionsPattern[]] :=
	GPUDeclareIndexBlock[GetAndCheckOption[iGPUExpression, {opts}, "APITarget"], IntegerDimension[d], pitch, depth]

iGPUExpression[arg_, opts:OptionsPattern[]] :=
	If[MemberQ[Names["SymbolicC`C*"], ToString[Head[arg]]],
		arg,
		If[NumericQ[arg] && TrueQ[GetAndCheckOption[GPUType, {opts}, "TargetPrecision"]],
			GenerateCode[CExpression[arg]] <> "f",
			CExpression[arg]
		]
	]
	
End[]


EndPackage[]
