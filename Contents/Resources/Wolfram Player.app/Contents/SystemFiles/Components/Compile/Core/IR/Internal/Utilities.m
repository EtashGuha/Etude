
BeginPackage["Compile`Core`IR`Internal`Utilities`"]

IRFormatType

Begin["`Private`"]

IRFormatType[Undefined] := 
	"\[DottedSquare]"
	
IRFormatType[TypeSpecifier["Undefined"]] := 
	"U"
	
IRFormatType[TypeSpecifier["Uninitialized"]] := 
	"\[CapitalOSlash]"

IRFormatType[TypeSpecifier[args__]] :=
	"TypeSpecifier[" <> StringRiffle[args, ", "] <> "]"
IRFormatType[TypeSpecifier[args__ -> res_]] :=
	"TypeSpecifier[" <> StringRiffle[args, ", "] <> " -> " <> ToString[res] <> "]"
	
IRFormatType[t_] := 
	Module[ {ef = t["shortname"]},
		If[ StringQ[ef], ef, ToString[ef]]
	]


End[]
EndPackage[]