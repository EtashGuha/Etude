
BeginPackage["Compile`TypeSystem`Bootstrap`LibC`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]


(*
   LibC Functions
*)

"StaticAnalysisIgnore"[

setupTypes[st_] :=
	Module[{
		env = st["typeEnvironment"],
		extern = MetaData[<|"Linkage" -> "ExternalLibrary"|>]
	},

	env["declareType", TypeConstructor["Real80"]];


	env["declareFunction", Native`PrimitiveFunction["LibC`__cospi"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__cospi" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__cospif"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__cospif" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__sincospi_stret"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__sincospi_stret" |>]@
		TypeSpecifier[{"Real64"} -> "C`ConstantArray"["Real64", TypeLiteral[2, "Integer64"]]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__sincospif_stret"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__sincospif_stret" |>]@
		TypeSpecifier[{"Real32"} -> "C`ConstantArray"["Real32", TypeLiteral[2, "Integer64"]]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__sinpi"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__sinpi" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__sinpif"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__sinpif" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`abs"], 
		extern@MetaData[<| "ExternalFunctionName" -> "abs" |>]@
		TypeSpecifier[{"Integer32"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`access"], 
		extern@MetaData[<| "ExternalFunctionName" -> "access" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer32"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`acos"], 
		extern@MetaData[<| "ExternalFunctionName" -> "acos" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`acosf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "acosf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`acosh"], 
		extern@MetaData[<| "ExternalFunctionName" -> "acosh" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`acoshf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "acoshf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`acoshl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "acoshl" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`acosl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "acosl" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`asin"], 
		extern@MetaData[<| "ExternalFunctionName" -> "asin" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`asinf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "asinf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`asinh"], 
		extern@MetaData[<| "ExternalFunctionName" -> "asinh" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`asinhf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "asinhf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`asinhl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "asinhl" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`asinl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "asinl" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`atan"], 
		extern@MetaData[<| "ExternalFunctionName" -> "atan" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`atan2"], 
		extern@MetaData[<| "ExternalFunctionName" -> "atan2" |>]@
		TypeSpecifier[{"Real64", "Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`atan2f"], 
		extern@MetaData[<| "ExternalFunctionName" -> "atan2f" |>]@
		TypeSpecifier[{"Real32", "Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`atan2l"], 
		extern@MetaData[<| "ExternalFunctionName" -> "atan2l" |>]@
		TypeSpecifier[{"Real80", "Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`atanf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "atanf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`atanh"], 
		extern@MetaData[<| "ExternalFunctionName" -> "atanh" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`atanhf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "atanhf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`atanhl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "atanhl" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`atanl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "atanl" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`atof"], 
		extern@MetaData[<| "ExternalFunctionName" -> "atof" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`atoi"], 
		extern@MetaData[<| "ExternalFunctionName" -> "atoi" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`atol"], 
		extern@MetaData[<| "ExternalFunctionName" -> "atol" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`atoll"], 
		extern@MetaData[<| "ExternalFunctionName" -> "atoll" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`bcmp"], 
		extern@MetaData[<| "ExternalFunctionName" -> "bcmp" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "Integer64"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`bcopy"], 
		extern@MetaData[<| "ExternalFunctionName" -> "bcopy" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "Integer64"} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`bzero"], 
		extern@MetaData[<| "ExternalFunctionName" -> "bzero" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer64"} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`calloc"], 
		extern@MetaData[<| "ExternalFunctionName" -> "calloc" |>]@
		TypeSpecifier[{"Integer64", "Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`cbrt"], 
		extern@MetaData[<| "ExternalFunctionName" -> "cbrt" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`cbrtf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "cbrtf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`cbrtl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "cbrtl" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`ceil"], 
		extern@MetaData[<| "ExternalFunctionName" -> "ceil" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`ceilf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "ceilf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`ceill"], 
		extern@MetaData[<| "ExternalFunctionName" -> "ceill" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`chown"], 
		extern@MetaData[<| "ExternalFunctionName" -> "chown" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer32", "Integer32"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`clearerr"], 
		extern@MetaData[<| "ExternalFunctionName" -> "clearerr" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`copysign"], 
		extern@MetaData[<| "ExternalFunctionName" -> "copysign" |>]@
		TypeSpecifier[{"Real64", "Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`copysignf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "copysignf" |>]@
		TypeSpecifier[{"Real32", "Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`copysignl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "copysignl" |>]@
		TypeSpecifier[{"Real80", "Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`cos"], 
		extern@MetaData[<| "ExternalFunctionName" -> "cos" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`cosf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "cosf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`cosh"], 
		extern@MetaData[<| "ExternalFunctionName" -> "cosh" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`coshf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "coshf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`coshl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "coshl" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`cosl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "cosl" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`ctermid"], 
		extern@MetaData[<| "ExternalFunctionName" -> "ctermid" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`exp"], 
		extern@MetaData[<| "ExternalFunctionName" -> "exp" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`exp2"], 
		extern@MetaData[<| "ExternalFunctionName" -> "exp2" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`exp2f"], 
		extern@MetaData[<| "ExternalFunctionName" -> "exp2f" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`exp2l"], 
		extern@MetaData[<| "ExternalFunctionName" -> "exp2l" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`expf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "expf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`expl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "expl" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`expm1"], 
		extern@MetaData[<| "ExternalFunctionName" -> "expm1" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`expm1f"], 
		extern@MetaData[<| "ExternalFunctionName" -> "expm1f" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`expm1l"], 
		extern@MetaData[<| "ExternalFunctionName" -> "expm1l" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fabs"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fabs" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fabsf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fabsf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fabsl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fabsl" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fclose"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fclose" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`feof"], 
		extern@MetaData[<| "ExternalFunctionName" -> "feof" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`ferror"], 
		extern@MetaData[<| "ExternalFunctionName" -> "ferror" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fflush"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fflush" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`ffs"], 
		extern@MetaData[<| "ExternalFunctionName" -> "ffs" |>]@
		TypeSpecifier[{"Integer32"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`ffsl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "ffsl" |>]@
		TypeSpecifier[{"Integer64"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`ffsll"], 
		extern@MetaData[<| "ExternalFunctionName" -> "ffsll" |>]@
		TypeSpecifier[{"Integer64"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fgetc"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fgetc" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fgetpos"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fgetpos" |>]@
		TypeSpecifier[{"VoidHandle", "Handle"["Integer64"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fgets"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fgets" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer32", "VoidHandle"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fileno"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fileno" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`flockfile"], 
		extern@MetaData[<| "ExternalFunctionName" -> "flockfile" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`floor"], 
		extern@MetaData[<| "ExternalFunctionName" -> "floor" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`floorf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "floorf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`floorl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "floorl" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fls"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fls" |>]@
		TypeSpecifier[{"Integer32"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`flsl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "flsl" |>]@
		TypeSpecifier[{"Integer64"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`flsll"], 
		extern@MetaData[<| "ExternalFunctionName" -> "flsll" |>]@
		TypeSpecifier[{"Integer64"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fmax"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fmax" |>]@
		TypeSpecifier[{"Real64", "Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fmaxf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fmaxf" |>]@
		TypeSpecifier[{"Real32", "Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fmaxl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fmaxl" |>]@
		TypeSpecifier[{"Real80", "Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fmin"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fmin" |>]@
		TypeSpecifier[{"Real64", "Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fminf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fminf" |>]@
		TypeSpecifier[{"Real32", "Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fminl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fminl" |>]@
		TypeSpecifier[{"Real80", "Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fmod"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fmod" |>]@
		TypeSpecifier[{"Real64", "Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fmodf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fmodf" |>]@
		TypeSpecifier[{"Real32", "Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fmodl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fmodl" |>]@
		TypeSpecifier[{"Real80", "Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fprintf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fprintf" |>]@
		TypeSpecifier[{"VoidHandle", "CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fputc"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fputc" |>]@
		TypeSpecifier[{"Integer32", "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fread"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fread" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer64", "Integer64", "VoidHandle"} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`free"], 
		extern@MetaData[<| "ExternalFunctionName" -> "free", "SandboxAllowed" -> True |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`frexp"], 
		extern@MetaData[<| "ExternalFunctionName" -> "frexp" |>]@
		TypeSpecifier[{"Real64", "Handle"["Integer32"]} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`frexpf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "frexpf" |>]@
		TypeSpecifier[{"Real32", "Handle"["Integer32"]} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`frexpl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "frexpl" |>]@
		TypeSpecifier[{"Real80", "Handle"["Integer32"]} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fscanf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fscanf" |>]@
		TypeSpecifier[{"VoidHandle", "CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fseek"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fseek" |>]@
		TypeSpecifier[{"VoidHandle", "Integer64", "Integer32"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fseeko"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fseeko" |>]@
		TypeSpecifier[{"VoidHandle", "Integer64", "Integer32"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fsetpos"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fsetpos" |>]@
		TypeSpecifier[{"VoidHandle", "Handle"["Integer64"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fstatvfs"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fstatvfs" |>]@
		TypeSpecifier[{"Integer32", "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`ftell"], 
		extern@MetaData[<| "ExternalFunctionName" -> "ftell" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`ftello"], 
		extern@MetaData[<| "ExternalFunctionName" -> "ftello" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`ftrylockfile"], 
		extern@MetaData[<| "ExternalFunctionName" -> "ftrylockfile" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`funlockfile"], 
		extern@MetaData[<| "ExternalFunctionName" -> "funlockfile" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`getc"], 
		extern@MetaData[<| "ExternalFunctionName" -> "getc" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`getc_unlocked"], 
		extern@MetaData[<| "ExternalFunctionName" -> "getc_unlocked" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`getchar"], 
		extern@MetaData[<| "ExternalFunctionName" -> "getchar" |>]@
		TypeSpecifier[{} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`getenv"], 
		extern@MetaData[<| "ExternalFunctionName" -> "getenv" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`getitimer"], 
		extern@MetaData[<| "ExternalFunctionName" -> "getitimer" |>]@
		TypeSpecifier[{"Integer32", "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`getlogin_r"], 
		extern@MetaData[<| "ExternalFunctionName" -> "getlogin_r" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer64"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`getpwnam"], 
		extern@MetaData[<| "ExternalFunctionName" -> "getpwnam" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "VoidHandle"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`gets"], 
		extern@MetaData[<| "ExternalFunctionName" -> "gets" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`gettimeofday"], 
		extern@MetaData[<| "ExternalFunctionName" -> "gettimeofday" |>]@
		TypeSpecifier[{"VoidHandle", "CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`_Z7isasciii"], 
		extern@MetaData[<| "ExternalFunctionName" -> "_Z7isasciii" |>]@
		TypeSpecifier[{"Integer32"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`_Z7isdigiti"], 
		extern@MetaData[<| "ExternalFunctionName" -> "_Z7isdigiti" |>]@
		TypeSpecifier[{"Integer32"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`labs"], 
		extern@MetaData[<| "ExternalFunctionName" -> "labs" |>]@
		TypeSpecifier[{"Integer64"} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`ldexp"], 
		extern@MetaData[<| "ExternalFunctionName" -> "ldexp" |>]@
		TypeSpecifier[{"Real64", "Integer32"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`ldexpf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "ldexpf" |>]@
		TypeSpecifier[{"Real32", "Integer32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`ldexpl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "ldexpl" |>]@
		TypeSpecifier[{"Real80", "Integer32"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`llabs"], 
		extern@MetaData[<| "ExternalFunctionName" -> "llabs" |>]@
		TypeSpecifier[{"Integer64"} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`log"], 
		extern@MetaData[<| "ExternalFunctionName" -> "log" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`log10"], 
		extern@MetaData[<| "ExternalFunctionName" -> "log10" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`log10f"], 
		extern@MetaData[<| "ExternalFunctionName" -> "log10f" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`log10l"], 
		extern@MetaData[<| "ExternalFunctionName" -> "log10l" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`log1p"], 
		extern@MetaData[<| "ExternalFunctionName" -> "log1p" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`log1pf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "log1pf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`log1pl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "log1pl" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`log2"], 
		extern@MetaData[<| "ExternalFunctionName" -> "log2" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`log2f"], 
		extern@MetaData[<| "ExternalFunctionName" -> "log2f" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`log2l"], 
		extern@MetaData[<| "ExternalFunctionName" -> "log2l" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`logb"], 
		extern@MetaData[<| "ExternalFunctionName" -> "logb" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`logbf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "logbf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`logbl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "logbl" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`logf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "logf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`logl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "logl" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`malloc"], 
		extern@MetaData[<| "ExternalFunctionName" -> "malloc", "SandboxAllowed" -> True |>]@
		TypeSpecifier[{"Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`memccpy"], 
		extern@MetaData[<| "ExternalFunctionName" -> "memccpy" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "Integer32", "Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`memchr"], 
		extern@MetaData[<| "ExternalFunctionName" -> "memchr" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer32", "Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`memcmp"], 
		extern@MetaData[<| "ExternalFunctionName" -> "memcmp" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "Integer64"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`memcpy"], 
		extern@MetaData[<| "ExternalFunctionName" -> "memcpy" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`memmove"], 
		extern@MetaData[<| "ExternalFunctionName" -> "memmove" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`memset"], 
		extern@MetaData[<| "ExternalFunctionName" -> "memset" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer32", "Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`memset_pattern16"], 
		extern@MetaData[<| "ExternalFunctionName" -> "memset_pattern16" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "Integer64"} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`mkdir"], 
		extern@MetaData[<| "ExternalFunctionName" -> "mkdir" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer16"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`modf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "modf" |>]@
		TypeSpecifier[{"Real64", "Handle"["Real64"]} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`modff"], 
		extern@MetaData[<| "ExternalFunctionName" -> "modff" |>]@
		TypeSpecifier[{"Real32", "Handle"["Real32"]} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`modfl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "modfl" |>]@
		TypeSpecifier[{"Real80", "Handle"["Real80"]} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`nearbyint"], 
		extern@MetaData[<| "ExternalFunctionName" -> "nearbyint" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`nearbyintf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "nearbyintf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`nearbyintl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "nearbyintl" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`pclose"], 
		extern@MetaData[<| "ExternalFunctionName" -> "pclose" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`perror"], 
		extern@MetaData[<| "ExternalFunctionName" -> "perror" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`posix_memalign"], 
		extern@MetaData[<| "ExternalFunctionName" -> "posix_memalign" |>]@
		TypeSpecifier[{"Handle"["CArray"["C`char"]], "Integer64", "Integer64"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`pow"], 
		extern@MetaData[<| "ExternalFunctionName" -> "pow" |>]@
		TypeSpecifier[{"Real64", "Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`powf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "powf" |>]@
		TypeSpecifier[{"Real32", "Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`powl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "powl" |>]@
		TypeSpecifier[{"Real80", "Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`printf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "printf" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`putc"], 
		extern@MetaData[<| "ExternalFunctionName" -> "putc" |>]@
		TypeSpecifier[{"Integer32", "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`putchar"], 
		extern@MetaData[<| "ExternalFunctionName" -> "putchar" |>]@
		TypeSpecifier[{"Integer32"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`puts"], 
		extern@MetaData[<| "ExternalFunctionName" -> "puts" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`qsort"], 
		extern@MetaData[<| "ExternalFunctionName" -> "qsort" |>]@
		TypeSpecifier[{"VoidHandle", "C`size_t", "C`size_t", {"VoidHandle", "VoidHandle"} -> "C`int"} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`readlink"], 
		extern@MetaData[<| "ExternalFunctionName" -> "readlink" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "Integer64"} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`realloc"], 
		extern@MetaData[<| "ExternalFunctionName" -> "realloc" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`reallocf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "reallocf" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`remove"], 
		extern@MetaData[<| "ExternalFunctionName" -> "remove" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`rename"], 
		extern@MetaData[<| "ExternalFunctionName" -> "rename" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`rewind"], 
		extern@MetaData[<| "ExternalFunctionName" -> "rewind" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`rint"], 
		extern@MetaData[<| "ExternalFunctionName" -> "rint" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`rintf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "rintf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`rintl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "rintl" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`rmdir"], 
		extern@MetaData[<| "ExternalFunctionName" -> "rmdir" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`round"], 
		extern@MetaData[<| "ExternalFunctionName" -> "round" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`roundf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "roundf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`roundl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "roundl" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`scanf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "scanf" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`setbuf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "setbuf" |>]@
		TypeSpecifier[{"VoidHandle", "CArray"["C`char"]} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`setitimer"], 
		extern@MetaData[<| "ExternalFunctionName" -> "setitimer" |>]@
		TypeSpecifier[{"Integer32", "VoidHandle", "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`setvbuf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "setvbuf" |>]@
		TypeSpecifier[{"VoidHandle", "CArray"["C`char"], "Integer32", "Integer64"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`sin"], 
		extern@MetaData[<| "ExternalFunctionName" -> "sin" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`sinf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "sinf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`sinh"], 
		extern@MetaData[<| "ExternalFunctionName" -> "sinh" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`sinhf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "sinhf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`sinhl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "sinhl" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`sinl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "sinl" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`snprintf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "snprintf" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer64", "CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`sprintf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "sprintf" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`sqrt"], 
		extern@MetaData[<| "ExternalFunctionName" -> "sqrt" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`sqrtf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "sqrtf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`sqrtl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "sqrtl" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`sscanf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "sscanf" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`statvfs"], 
		extern@MetaData[<| "ExternalFunctionName" -> "statvfs" |>]@
		TypeSpecifier[{"CArray"["C`char"], "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`stpcpy"], 
		extern@MetaData[<| "ExternalFunctionName" -> "stpcpy" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"]} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`stpncpy"], 
		extern@MetaData[<| "ExternalFunctionName" -> "stpncpy" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strcasecmp"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strcasecmp" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strcat"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strcat" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"]} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strchr"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strchr" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer32"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strcmp"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strcmp" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strcoll"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strcoll" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strcpy"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strcpy" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"]} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strcspn"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strcspn" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"]} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strdup"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strdup" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strlen"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strlen" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strncasecmp"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strncasecmp" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "Integer64"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strncat"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strncat" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strncmp"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strncmp" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "Integer64"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strncpy"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strncpy" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strndup"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strndup" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strnlen"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strnlen" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer64"} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strpbrk"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strpbrk" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"]} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strrchr"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strrchr" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer32"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strspn"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strspn" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"]} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strstr"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strstr" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"]} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strtok"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strtok" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"]} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strtok_r"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strtok_r" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "Handle"["CArray"["C`char"]]} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strtol"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strtol" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Handle"["CArray"["C`char"]], "Integer32"} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strtold"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strtold" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Handle"["CArray"["C`char"]]} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strtoll"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strtoll" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Handle"["CArray"["C`char"]], "Integer32"} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strtoul"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strtoul" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Handle"["CArray"["C`char"]], "Integer32"} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strtoull"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strtoull" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Handle"["CArray"["C`char"]], "Integer32"} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strxfrm"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strxfrm" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "Integer64"} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`tan"], 
		extern@MetaData[<| "ExternalFunctionName" -> "tan" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`tanf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "tanf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`tanh"], 
		extern@MetaData[<| "ExternalFunctionName" -> "tanh" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`tanhf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "tanhf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`tanhl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "tanhl" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`tanl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "tanl" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`times"], 
		extern@MetaData[<| "ExternalFunctionName" -> "times" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`tmpfile"], 
		extern@MetaData[<| "ExternalFunctionName" -> "tmpfile" |>]@
		TypeSpecifier[{} -> "VoidHandle"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`_Z7toasciii"], 
		extern@MetaData[<| "ExternalFunctionName" -> "_Z7toasciii" |>]@
		TypeSpecifier[{"Integer32"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`trunc"], 
		extern@MetaData[<| "ExternalFunctionName" -> "trunc" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`truncf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "truncf" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`truncl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "truncl" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`uname"], 
		extern@MetaData[<| "ExternalFunctionName" -> "uname" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`ungetc"], 
		extern@MetaData[<| "ExternalFunctionName" -> "ungetc" |>]@
		TypeSpecifier[{"Integer32", "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`unlink"], 
		extern@MetaData[<| "ExternalFunctionName" -> "unlink" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`utime"], 
		extern@MetaData[<| "ExternalFunctionName" -> "utime" |>]@
		TypeSpecifier[{"CArray"["C`char"], "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`utimes"], 
		extern@MetaData[<| "ExternalFunctionName" -> "utimes" |>]@
		TypeSpecifier[{"CArray"["C`char"], "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`valloc"], 
		extern@MetaData[<| "ExternalFunctionName" -> "valloc" |>]@
		TypeSpecifier[{"Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`vfprintf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "vfprintf" |>]@
		TypeSpecifier[{"VoidHandle", "CArray"["C`char"], "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`vfscanf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "vfscanf" |>]@
		TypeSpecifier[{"VoidHandle", "CArray"["C`char"], "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`vprintf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "vprintf" |>]@
		TypeSpecifier[{"CArray"["C`char"], "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`vscanf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "vscanf" |>]@
		TypeSpecifier[{"CArray"["C`char"], "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`vsnprintf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "vsnprintf" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer64", "CArray"["C`char"], "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`vsprintf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "vsprintf" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`vsscanf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "vsscanf" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`wcslen"], 
		extern@MetaData[<| "ExternalFunctionName" -> "wcslen" |>]@
		TypeSpecifier[{"Handle"["Integer32"]} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`chmod"], 
		extern@MetaData[<| "ExternalFunctionName" -> "chmod" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer16"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`closedir"], 
		extern@MetaData[<| "ExternalFunctionName" -> "closedir" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fdopen"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fdopen" |>]@
		TypeSpecifier[{"Integer32", "CArray"["C`char"]} -> "VoidHandle"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fopen"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fopen" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"]} -> "VoidHandle"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fputs"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fputs" |>]@
		TypeSpecifier[{"CArray"["C`char"], "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fstat"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fstat" |>]@
		TypeSpecifier[{"Integer32", "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fwrite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fwrite" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer64", "Integer64", "VoidHandle"} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`lchown"], 
		extern@MetaData[<| "ExternalFunctionName" -> "lchown" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer32", "Integer32"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`lstat"], 
		extern@MetaData[<| "ExternalFunctionName" -> "lstat" |>]@
		TypeSpecifier[{"CArray"["C`char"], "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`mktime"], 
		extern@MetaData[<| "ExternalFunctionName" -> "mktime" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`open"], 
		extern@MetaData[<| "ExternalFunctionName" -> "open" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer32"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`opendir"], 
		extern@MetaData[<| "ExternalFunctionName" -> "opendir" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "VoidHandle"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`popen"], 
		extern@MetaData[<| "ExternalFunctionName" -> "popen" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"]} -> "VoidHandle"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`pread"], 
		extern@MetaData[<| "ExternalFunctionName" -> "pread" |>]@
		TypeSpecifier[{"Integer32", "CArray"["C`char"], "Integer64", "Integer64"} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`pwrite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "pwrite" |>]@
		TypeSpecifier[{"Integer32", "CArray"["C`char"], "Integer64", "Integer64"} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`read"], 
		extern@MetaData[<| "ExternalFunctionName" -> "read" |>]@
		TypeSpecifier[{"Integer32", "CArray"["C`char"], "Integer64"} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`realpath"], 
		extern@MetaData[<| "ExternalFunctionName" -> "realpath" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"]} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`stat"], 
		extern@MetaData[<| "ExternalFunctionName" -> "stat" |>]@
		TypeSpecifier[{"CArray"["C`char"], "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strtod"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strtod" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Handle"["CArray"["C`char"]]} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`strtof"], 
		extern@MetaData[<| "ExternalFunctionName" -> "strtof" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Handle"["CArray"["C`char"]]} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`system"], 
		extern@MetaData[<| "ExternalFunctionName" -> "system" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`unsetenv"], 
		extern@MetaData[<| "ExternalFunctionName" -> "unsetenv" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`write"], 
		extern@MetaData[<| "ExternalFunctionName" -> "write" |>]@
		TypeSpecifier[{"Integer32", "CArray"["C`char"], "Integer64"} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fopen64"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fopen64" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"]} -> "VoidHandle"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fstat64"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fstat64" |>]@
		TypeSpecifier[{"Integer32", "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fstatvfs64"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fstatvfs64" |>]@
		TypeSpecifier[{"Integer32", "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`lstat64"], 
		extern@MetaData[<| "ExternalFunctionName" -> "lstat64" |>]@
		TypeSpecifier[{"CArray"["C`char"], "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`open64"], 
		extern@MetaData[<| "ExternalFunctionName" -> "open64" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer32"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`stat64"], 
		extern@MetaData[<| "ExternalFunctionName" -> "stat64" |>]@
		TypeSpecifier[{"CArray"["C`char"], "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`statvfs64"], 
		extern@MetaData[<| "ExternalFunctionName" -> "statvfs64" |>]@
		TypeSpecifier[{"CArray"["C`char"], "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`tmpfile64"], 
		extern@MetaData[<| "ExternalFunctionName" -> "tmpfile64" |>]@
		TypeSpecifier[{} -> "VoidHandle"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fseeko64"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fseeko64" |>]@
		TypeSpecifier[{"VoidHandle", "Integer64", "Integer32"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`ftello64"], 
		extern@MetaData[<| "ExternalFunctionName" -> "ftello64" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Integer64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`_ZdaPv"], 
		extern@MetaData[<| "ExternalFunctionName" -> "_ZdaPv" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`_ZdaPvRKSt9nothrow_t"], 
		extern@MetaData[<| "ExternalFunctionName" -> "_ZdaPvRKSt9nothrow_t" |>]@
		TypeSpecifier[{"CArray"["C`char"], "VoidHandle"} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`_ZdaPvj"], 
		extern@MetaData[<| "ExternalFunctionName" -> "_ZdaPvj" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer32"} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`_ZdaPvm"], 
		extern@MetaData[<| "ExternalFunctionName" -> "_ZdaPvm" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer64"} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`_ZdlPv"], 
		extern@MetaData[<| "ExternalFunctionName" -> "_ZdlPv" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`_ZdlPvRKSt9nothrow_t"], 
		extern@MetaData[<| "ExternalFunctionName" -> "_ZdlPvRKSt9nothrow_t" |>]@
		TypeSpecifier[{"CArray"["C`char"], "VoidHandle"} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`_ZdlPvj"], 
		extern@MetaData[<| "ExternalFunctionName" -> "_ZdlPvj" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer32"} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`_ZdlPvm"], 
		extern@MetaData[<| "ExternalFunctionName" -> "_ZdlPvm" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer64"} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`_Znaj"], 
		extern@MetaData[<| "ExternalFunctionName" -> "_Znaj" |>]@
		TypeSpecifier[{"Integer32"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`_ZnajRKSt9nothrow_t"], 
		extern@MetaData[<| "ExternalFunctionName" -> "_ZnajRKSt9nothrow_t" |>]@
		TypeSpecifier[{"Integer32", "VoidHandle"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`_Znam"], 
		extern@MetaData[<| "ExternalFunctionName" -> "_Znam" |>]@
		TypeSpecifier[{"Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`_ZnamRKSt9nothrow_t"], 
		extern@MetaData[<| "ExternalFunctionName" -> "_ZnamRKSt9nothrow_t" |>]@
		TypeSpecifier[{"Integer64", "VoidHandle"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`_Znwj"], 
		extern@MetaData[<| "ExternalFunctionName" -> "_Znwj" |>]@
		TypeSpecifier[{"Integer32"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`_ZnwjRKSt9nothrow_t"], 
		extern@MetaData[<| "ExternalFunctionName" -> "_ZnwjRKSt9nothrow_t" |>]@
		TypeSpecifier[{"Integer32", "VoidHandle"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`_Znwm"], 
		extern@MetaData[<| "ExternalFunctionName" -> "_Znwm" |>]@
		TypeSpecifier[{"Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`_ZnwmRKSt9nothrow_t"], 
		extern@MetaData[<| "ExternalFunctionName" -> "_ZnwmRKSt9nothrow_t" |>]@
		TypeSpecifier[{"Integer64", "VoidHandle"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`??3@YAXPEAX@Z"], 
		extern@MetaData[<| "ExternalFunctionName" -> "??3@YAXPEAX@Z" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`??3@YAXPEAXAEBUnothrow_t@std@@@Z"], 
		extern@MetaData[<| "ExternalFunctionName" -> "??3@YAXPEAXAEBUnothrow_t@std@@@Z" |>]@
		TypeSpecifier[{"CArray"["C`char"], "VoidHandle"} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`??3@YAXPEAX_K@Z"], 
		extern@MetaData[<| "ExternalFunctionName" -> "??3@YAXPEAX_K@Z" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer64"} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`??_V@YAXPEAX@Z"], 
		extern@MetaData[<| "ExternalFunctionName" -> "??_V@YAXPEAX@Z" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`??_V@YAXPEAXAEBUnothrow_t@std@@@Z"], 
		extern@MetaData[<| "ExternalFunctionName" -> "??_V@YAXPEAXAEBUnothrow_t@std@@@Z" |>]@
		TypeSpecifier[{"CArray"["C`char"], "VoidHandle"} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`??_V@YAXPEAX_K@Z"], 
		extern@MetaData[<| "ExternalFunctionName" -> "??_V@YAXPEAX_K@Z" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer64"} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`??2@YAPAXI@Z"], 
		extern@MetaData[<| "ExternalFunctionName" -> "??2@YAPAXI@Z" |>]@
		TypeSpecifier[{"Integer32"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`??2@YAPAXIABUnothrow_t@std@@@Z"], 
		extern@MetaData[<| "ExternalFunctionName" -> "??2@YAPAXIABUnothrow_t@std@@@Z" |>]@
		TypeSpecifier[{"Integer32", "VoidHandle"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`??2@YAPEAX_K@Z"], 
		extern@MetaData[<| "ExternalFunctionName" -> "??2@YAPEAX_K@Z" |>]@
		TypeSpecifier[{"Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`??2@YAPEAX_KAEBUnothrow_t@std@@@Z"], 
		extern@MetaData[<| "ExternalFunctionName" -> "??2@YAPEAX_KAEBUnothrow_t@std@@@Z" |>]@
		TypeSpecifier[{"Integer64", "VoidHandle"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`??_U@YAPAXI@Z"], 
		extern@MetaData[<| "ExternalFunctionName" -> "??_U@YAPAXI@Z" |>]@
		TypeSpecifier[{"Integer32"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`??_U@YAPAXIABUnothrow_t@std@@@Z"], 
		extern@MetaData[<| "ExternalFunctionName" -> "??_U@YAPAXIABUnothrow_t@std@@@Z" |>]@
		TypeSpecifier[{"Integer32", "VoidHandle"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`??_U@YAPEAX_K@Z"], 
		extern@MetaData[<| "ExternalFunctionName" -> "??_U@YAPEAX_K@Z" |>]@
		TypeSpecifier[{"Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`??_U@YAPEAX_KAEBUnothrow_t@std@@@Z"], 
		extern@MetaData[<| "ExternalFunctionName" -> "??_U@YAPEAX_KAEBUnothrow_t@std@@@Z" |>]@
		TypeSpecifier[{"Integer64", "VoidHandle"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`??3@YAXPAX@Z"], 
		extern@MetaData[<| "ExternalFunctionName" -> "??3@YAXPAX@Z" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`??3@YAXPAXABUnothrow_t@std@@@Z"], 
		extern@MetaData[<| "ExternalFunctionName" -> "??3@YAXPAXABUnothrow_t@std@@@Z" |>]@
		TypeSpecifier[{"CArray"["C`char"], "VoidHandle"} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`??3@YAXPAXI@Z"], 
		extern@MetaData[<| "ExternalFunctionName" -> "??3@YAXPAXI@Z" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer32"} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`??_V@YAXPAX@Z"], 
		extern@MetaData[<| "ExternalFunctionName" -> "??_V@YAXPAX@Z" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`??_V@YAXPAXABUnothrow_t@std@@@Z"], 
		extern@MetaData[<| "ExternalFunctionName" -> "??_V@YAXPAXABUnothrow_t@std@@@Z" |>]@
		TypeSpecifier[{"CArray"["C`char"], "VoidHandle"} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`??_V@YAXPAXI@Z"], 
		extern@MetaData[<| "ExternalFunctionName" -> "??_V@YAXPAXI@Z" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer32"} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__cxa_atexit"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__cxa_atexit" |>]@
		TypeSpecifier[{{} -> "Void"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__cxa_guard_abort"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__cxa_guard_abort" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__cxa_guard_acquire"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__cxa_guard_acquire" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__cxa_guard_release"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__cxa_guard_release" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Void"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__nvvm_reflect"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__nvvm_reflect" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__memcpy_chk"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__memcpy_chk" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "Integer64", "Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__memmove_chk"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__memmove_chk" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "Integer64", "Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__memset_chk"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__memset_chk" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer32", "Integer64", "Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__stpcpy_chk"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__stpcpy_chk" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__stpncpy_chk"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__stpncpy_chk" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "Integer64", "Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__strcpy_chk"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__strcpy_chk" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__strncpy_chk"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__strncpy_chk" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "Integer64", "Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`memalign"], 
		extern@MetaData[<| "ExternalFunctionName" -> "memalign" |>]@
		TypeSpecifier[{"Integer64", "Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`mempcpy"], 
		extern@MetaData[<| "ExternalFunctionName" -> "mempcpy" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`memrchr"], 
		extern@MetaData[<| "ExternalFunctionName" -> "memrchr" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer32", "Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`_IO_getc"], 
		extern@MetaData[<| "ExternalFunctionName" -> "_IO_getc" |>]@
		TypeSpecifier[{"VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`_IO_putc"], 
		extern@MetaData[<| "ExternalFunctionName" -> "_IO_putc" |>]@
		TypeSpecifier[{"Integer32", "VoidHandle"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__isoc99_scanf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__isoc99_scanf" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__isoc99_sscanf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__isoc99_sscanf" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__strdup"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__strdup" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__strndup"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__strndup" |>]@
		TypeSpecifier[{"CArray"["C`char"], "Integer64"} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__strtok_r"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__strtok_r" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"], "Handle"["CArray"["C`char"]]} -> "CArray"["C`char"]]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__sqrt_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__sqrt_finite" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__sqrtf_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__sqrtf_finite" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__sqrtl_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__sqrtl_finite" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`exp10"], 
		extern@MetaData[<| "ExternalFunctionName" -> "exp10" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`exp10f"], 
		extern@MetaData[<| "ExternalFunctionName" -> "exp10f" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`exp10l"], 
		extern@MetaData[<| "ExternalFunctionName" -> "exp10l" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`fiprintf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "fiprintf" |>]@
		TypeSpecifier[{"VoidHandle", "CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`iprintf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "iprintf" |>]@
		TypeSpecifier[{"CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`siprintf"], 
		extern@MetaData[<| "ExternalFunctionName" -> "siprintf" |>]@
		TypeSpecifier[{"CArray"["C`char"], "CArray"["C`char"]} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`htonl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "htonl" |>]@
		TypeSpecifier[{"Integer32"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`htons"], 
		extern@MetaData[<| "ExternalFunctionName" -> "htons" |>]@
		TypeSpecifier[{"Integer16"} -> "Integer16"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`ntohl"], 
		extern@MetaData[<| "ExternalFunctionName" -> "ntohl" |>]@
		TypeSpecifier[{"Integer32"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`ntohs"], 
		extern@MetaData[<| "ExternalFunctionName" -> "ntohs" |>]@
		TypeSpecifier[{"Integer16"} -> "Integer16"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`isascii"], 
		extern@MetaData[<| "ExternalFunctionName" -> "isascii" |>]@
		TypeSpecifier[{"Integer32"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`isdigit"], 
		extern@MetaData[<| "ExternalFunctionName" -> "isdigit" |>]@
		TypeSpecifier[{"Integer32"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`toascii"], 
		extern@MetaData[<| "ExternalFunctionName" -> "toascii" |>]@
		TypeSpecifier[{"Integer32"} -> "Integer32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__acos_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__acos_finite" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__acosf_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__acosf_finite" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__acosl_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__acosl_finite" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__acosh_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__acosh_finite" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__acoshf_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__acoshf_finite" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__acoshl_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__acoshl_finite" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__asin_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__asin_finite" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__asinf_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__asinf_finite" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__asinl_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__asinl_finite" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__atan2_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__atan2_finite" |>]@
		TypeSpecifier[{"Real64", "Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__atan2f_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__atan2f_finite" |>]@
		TypeSpecifier[{"Real32", "Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__atan2l_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__atan2l_finite" |>]@
		TypeSpecifier[{"Real80", "Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__atanh_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__atanh_finite" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__atanhf_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__atanhf_finite" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__atanhl_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__atanhl_finite" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__cosh_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__cosh_finite" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__coshf_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__coshf_finite" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__coshl_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__coshl_finite" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__exp10_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__exp10_finite" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__exp10f_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__exp10f_finite" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__exp10l_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__exp10l_finite" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__exp2_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__exp2_finite" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__exp2f_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__exp2f_finite" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__exp2l_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__exp2l_finite" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__exp_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__exp_finite" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__expf_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__expf_finite" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__expl_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__expl_finite" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__log10_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__log10_finite" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__log10f_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__log10f_finite" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__log10l_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__log10l_finite" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__log2_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__log2_finite" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__log2f_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__log2f_finite" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__log2l_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__log2l_finite" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__log_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__log_finite" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__logf_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__logf_finite" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__logl_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__logl_finite" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__pow_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__pow_finite" |>]@
		TypeSpecifier[{"Real64", "Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__powf_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__powf_finite" |>]@
		TypeSpecifier[{"Real32", "Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__powl_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__powl_finite" |>]@
		TypeSpecifier[{"Real80", "Real80"} -> "Real80"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__sinh_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__sinh_finite" |>]@
		TypeSpecifier[{"Real64"} -> "Real64"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__sinhf_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__sinhf_finite" |>]@
		TypeSpecifier[{"Real32"} -> "Real32"]
	];
	env["declareFunction", Native`PrimitiveFunction["LibC`__sinhl_finite"], 
		extern@MetaData[<| "ExternalFunctionName" -> "__sinhl_finite" |>]@
		TypeSpecifier[{"Real80"} -> "Real80"]
	];
]

] (* StaticAnalysisIgnore *)


RegisterCallback["SetupTypeSystem", setupTypes]


End[]

EndPackage[]
