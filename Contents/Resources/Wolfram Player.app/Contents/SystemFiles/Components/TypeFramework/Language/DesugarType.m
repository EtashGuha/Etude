
BeginPackage["TypeFramework`Language`DesugarType`"]

DesugarType




Begin["`Private`"]

Needs["TypeFramework`"]
Needs["TypeFramework`TypeObjects`Kind`"]
Needs["CompileUtilities`Error`Exceptions`"]


desugarKind[ "*"] :=
	CreateNullaryKind[]
	
desugarKind[ {argsSeq__} -> retIn_ ] :=
	Module[ {args, ret},
		args = Map[ desugarKind, {argsSeq}];
		ret = desugarKind[ retIn];
		CreateFunctionKind[ args, ret]
	]
	

(*
MetaData
*)

DesugarType[ MetaData[data0_][MetaData[data1_][h_[args__]]]] :=
	DesugarType[ h[args, MetaData -> Join[data0, data1]]]
	
DesugarType[ MetaData[data_][h_[args__]]] :=
	DesugarType[ h[args, MetaData -> data]]



(*
  TypeConstructor
*)

DesugarType[ TypeConstructor[name_String, opts:OptionsPattern[]]] :=
	DesugarType[ TypeConstructor[name, "*", opts]]

DesugarType[ TypeConstructor[name_String, kind_, opts:OptionsPattern[]]] :=
	With[ {kindObj = desugarKind[kind]},
		TypeConstructor[name, kindObj, Association[opts]]
	]

(*
  TypeAlias
*)

DesugarType[ TypeAlias[name_String, type_, opts:OptionsPattern[]]] :=
    TypeAlias[DesugarType[TypeConstructor[name, opts]], type, <|opts|>];
    
DesugarType[ TypeAlias[a_, type_, opts:OptionsPattern[]]] :=
    TypeAlias[TypeSpecifier[a], type, <|opts|>];

(*
  AbstractType
*)

(*DesugarType[ AbstractType[name_String, vars_, types_, opts:OptionsPattern[]]] :=
	DesugarType[ AbstractType[name, vars, types, <||>, "*", opts]]

DesugarType[ AbstractType[name_String, vars_, types_, methods_, opts:OptionsPattern[]]] :=
	DesugarType[ AbstractType[name, vars, types, methods, "*", opts]]

DesugarType[ AbstractType[name_String, vars_, typesIn_, methodsIn_, kind_, opts:OptionsPattern[]]] :=
	Module[ {
			kindObj = desugarKind[kind], 
			deriving = Lookup[{opts},  "Deriving", {}],
			types = Map[ getType[ vars, #]&, typesIn],
			methods = Map[ getMethod[ vars, #]&, methodsIn]
		},
		CreateAbstractType[name, types, methods, deriving, kindObj]
	]


getType[ vars_, (Rule|RuleDelayed)[ fun_, ty_]] :=
	<|"variables" -> vars, "function" -> fun, "type" -> ty|>
	
getMethod[ vars_, (Rule|RuleDelayed)[ fun_, body_]] :=
	<|"variables" -> vars, "function" -> fun, "body" -> body|>
*)	

options = {"Deriving", "Default", "Constraints"}

checkHead[ (Rule|RuleDelayed)[h_, val_]] :=
	MemberQ[options, h] 
	
checkHead[_] :=
	False
	
checkOptions[opts_] :=
	AllTrue[Flatten[{opts}], checkHead]

DesugarType[ AbstractType[name_String]] :=
	DesugarType[ AbstractType[name, {}, {}, "*", {}]]

DesugarType[ AbstractType[name_String, opts__?checkOptions]] :=
	DesugarType[ AbstractType[name, {}, {}, "*", opts]]

DesugarType[ AbstractType[name_String, vars_]] :=
	DesugarType[ AbstractType[name, vars, {}, "*", {}]]

DesugarType[ AbstractType[name_String, vars_, opts__?checkOptions]] :=
	DesugarType[ AbstractType[name, vars, {}, "*", opts]]

DesugarType[ AbstractType[name_String, vars_, types_]] :=
	DesugarType[ AbstractType[name, vars, types, "*", {}]]

DesugarType[ AbstractType[name_String, vars_, types_, opts__?checkOptions]] :=
	DesugarType[ AbstractType[name, vars, types, "*", opts]]

DesugarType[ AbstractType[name_String, vars_, types_, kind_]] :=
	DesugarType[ AbstractType[name, vars, types, "*", {}]]

DesugarType[ AbstractType[name_String, vars_, data_, kind_, opts__?checkOptions]] :=
	With[{
        kindObject = desugarKind[kind]
	},
		AbstractType[name, vars, data, kindObject, opts]
	]



(*
  TypeInstance
*)

DesugarType[ TypeInstance[class_String, type_]] :=
	TypeInstance[class, {}, type, {}, <||>]
	
DesugarType[ TypeInstance[class_String, type_, opts__?checkOptions]] :=
	TypeInstance[class, {}, type, {}, <||>]
	
DesugarType[ TypeInstance[class_String, type_, funs_?ListQ]] /; !ListQ[type] :=
    TypeInstance[class, {}, type, {}, <||>]

DesugarType[ TypeInstance[class_String, type_, funs_?ListQ, opts__?checkOptions]] /; !ListQ[type] :=
    TypeInstance[class, {}, type, {}, Association[opts]]

DesugarType[ TypeInstance[class_String, vars_, type_]] :=
	TypeInstance[class, vars, type, {}, <||>]

DesugarType[ TypeInstance[class_String, vars_, type_, opts__?checkOptions]] :=
	TypeInstance[class, vars, type, {}, Association[opts]]

	
DesugarType[ TypeInstance[class_String, vars_, type_, funs_?ListQ]] :=
    TypeInstance[class, vars, type, funs, <||>]

DesugarType[ TypeInstance[class_String, vars_, type_, funs_,  opts__?checkOptions]] :=
	TypeInstance[class, vars, type, funs, Association[opts]]


DesugarType[args___] :=
	ThrowException[TypeInferenceException[{"Unknown call to DesugarType", {args}}]]

End[]

EndPackage[]

