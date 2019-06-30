
BeginPackage["SymbolicC`SymbolicCXX`"]


SymbolicCXXCommentLine::usage = "SymbolicCXXCommentLine[comment] is the FullForm of the C++ comment \"// comment\"."
SymbolicCXXTemplateInstance::usage = "SymbolicCXXTemplateInstance[name, {t1, t2, t3}] is the FullForm of the C++ expression \"name<t1, t2, t3>\"."

Begin["`Private`"]

Needs["SymbolicC`"]

CPrecedence[_SymbolicCXXTemplateInstance] = CPrecedence[CCall] 

SymbolicC`Private`IsCExpression[ _SymbolicCXXTemplateInstance] := True
SymbolicC`Private`IsCExpression[ _SymbolicCXXCommentLine] := True


GenerateCode[SymbolicCXXTemplateInstance[name_, type_ /; !ListQ[type]], opts:OptionsPattern[]] :=
	GenerateCode[SymbolicCXXTemplateInstance[name, {type}], opts]

GenerateCode[SymbolicCXXTemplateInstance[name_, types_List], opts:OptionsPattern[]] := 
	StringJoin[{
		GenerateCode[name, opts],
		"<",
		Riffle[GenerateCode[#, opts]& /@ types, ", "],
		">"
	}]

	
GenerateCode[SymbolicCXXCommentLine[x_], opts:OptionsPattern[]] := "// " <> ToString[x]	
    
End[]

EndPackage[]