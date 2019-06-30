
BeginPackage["Compile`TypeSystem`Bootstrap`C`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]


"StaticAnalysisIgnore"[

setupTypes[st_] :=
	Module[ {env = st["typeEnvironment"]},

		env["declareType", TypeConstructor["CArray", {"*"} -> "*", "Implements" -> "Indexable"]];
		env["declareType", TypeConstructor["C`ConstantArray", {"*", "*"} -> "*"]]; (* First argument is the type and the second is the length *)
		
		env["declareType", TypeAlias["C`size_t", "UnsignedInteger64"]];
		
		env["declareType", TypeAlias["C`char", "UnsignedInteger8"]];
		env["declareType", TypeAlias["C`short", "UnsignedInteger16"]];
		env["declareType", TypeAlias["C`int", "Integer32"]];
		env["declareType", TypeAlias["C`uint", "UnsignedInteger32"]];
		
		env["declareType", TypeAlias["C`int32", "Integer32"]];
		env["declareType", TypeAlias["C`int64", "Integer64"]];
		env["declareType", TypeAlias["C`uint32", "UnsignedInteger32"]];
		env["declareType", TypeAlias["C`uint64", "UnsignedInteger64"]];
		
		env["declareType", TypeAlias["C`half", "Real16"]];
		env["declareType", TypeAlias["C`float", "Real32"]];
		env["declareType", TypeAlias["C`double", "Real64"]];
		
		env["declareType", TypeAlias["CString", "CArray"["C`char"]]];
		
		env["declareFunction", Native`Equal, 
			Typed[{"CString", "CString"} -> "Boolean"
			]@Function[{c1, c2},
				Native`UncheckedBlock@Module[ { pos = 0},
					While[c1[[pos]] === c2[[pos]],
						If[c1[[pos]] === Typed[0, "C`char"],
								Return[True]];
						pos++];
					False
				]
			]];

		
		
	]

] (* StaticAnalysisIgnore *)

RegisterCallback["SetupTypeSystem", setupTypes]


End[]

EndPackage[]
