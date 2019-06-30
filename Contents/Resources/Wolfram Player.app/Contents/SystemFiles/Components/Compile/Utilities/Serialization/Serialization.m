
(*
  Implementation of serialization and driver for cloning.
  
  The two share a lot of functionality,  but have different drivers which 
  abstract some of the differences.  There are a few places where isClone
  is used and I suspect that these could be eliminated by abstracting the
  usages.
*)
BeginPackage["Compile`Utilities`Serialization`"]



WIRSerialize

WIRDeserialize

WIRSerialization

CreateCloneEnvironment

GetSerializationData

Begin["`Private`"]

Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`IR`ProgramModule`"]
Needs["Compile`Core`IR`FunctionInformation`"] (* For DeserializeFunctionInformation *)
Needs["Compile`Core`IR`FunctionInlineInformation`"] (* For DeserializeFunctionInlineInformation *)
Needs["CompileAST`Create`Construct`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionTraits`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]
Needs["Compile`Core`IR`Instruction`InstructionQ`"]


$SerializationVersion = 1
WIRDeserialize::versnf = "Unknown serialization metadata `1` has been encountered.  Valid data is expected to be an Association with a Version key."
WIRDeserialize::vers = "`1` is not a supported serialization version.  Version 1 is supported."


WIRSerialize[tyEnv_?TypeEnvironmentQ, arg_] :=
	CatchException[
		Module[ {env = CreateDeserializationEnvironment[tyEnv], ser},
			If[ !serializableQ[arg],
				ThrowException[CompilerException[{"Not serializable ", arg}]]
			];
			ser = arg["serialize", env];
			WIRSerialization[ ser, <| "Version" -> $SerializationVersion |>]
		]
		,
		{{_, CreateFailure}}
	]

serializableQ[arg_]	:=
	VariableQ[arg] || ConstantValueQ[arg] || InstructionQ[arg] || BasicBlockQ[arg] || FunctionModuleQ[arg] || ProgramModuleQ[arg]


checkVersion[ data_] :=
	Which[
		!KeyExistsQ[data, "Version"],
			Message[WIRDeserialize::versnf, data];
			False,
		data["Version"] =!= $SerializationVersion,
			Message[WIRDeserialize::vers, data["Version"]];
			False,
		True,
			True
	]


Options[ WIRDeserialize] =
	{"UniqueID" -> False}

WIRDeserialize[tyEnv_?TypeEnvironmentQ, WIRSerialization[ arg_, data_], opts:OptionsPattern[]] :=
	CatchException[
		Module[ {env = CreateDeserializationEnvironment[tyEnv, <|"uniqueID" -> OptionValue["UniqueID"]|>]},
			If[ !checkVersion[data],
				ThrowException[CompilerException[{"No Version", data}]]
				,
				env["deserialize", arg]
			]
		]
		,
		{{_, CreateFailure}}
	]

WIRDeserialize[args___] :=
	ThrowException[{"Unrecognized call to WIRDeserialize", {args}}]


$deserializationFunctions = 
<|
	"Variable" -> DeserializeVariable,
	"ConstantValue" -> DeserializeConstantValue,
	"Instruction" -> DeserializeInstruction,
	"BasicBlock" -> DeserializeBasicBlock,
	"FunctionModule" -> DeserializeFunctionModule,
    "FunctionInformation" -> DeserializeFunctionInformation,
    "FunctionInlineInformation" -> DeserializeFunctionInlineInformation,
	"ProgramModule" -> DeserializeProgramModule
|>

deserialize[self_, arg_] :=
	Module[ {fun},
		fun = Lookup[ $deserializationFunctions, Head[arg], Null];
		If[ fun === Null,
			ThrowException[CompilerException[{"Cannot deserialize argument", arg}]]
		];
		fun[self, arg]
	]

getElement[self_, data_, name_] :=
	Module[ {elem},
		elem = Lookup[data, name];
		If[MissingQ[elem],
			ThrowException[CompilerException[{"Cannot find expected argument ", name, data}]]
		];
		self["deserialize", elem]
	]



getElementList[self_, data_, name_] :=
	Module[ {elems},
		elems = Lookup[data, name];
		If[MissingQ[elems],
			ThrowException[CompilerException[{"Cannot find expected argument ", name, data}]]
		];
		If[!ListQ[elems],
			ThrowException[CompilerException[{"Argument is expected to be a list ", name, data}]]
		];
		Map[ self["deserialize",#]&, elems]
	]
	
getElementNoDeserialize[self_, data_, name_] :=
	Module[ {elem},
		elem = Lookup[data, name];
		If[MissingQ[elem],
			ThrowException[CompilerException[{"Cannot find expected argument ", name, data}]]
		];
		elem
	]
	

getElementMExpr[self_, data_, name_] :=
	Module[ {elem},
		elem = Lookup[data, name];
		If[MissingQ[elem],
			ThrowException[CompilerException[{"Cannot find expected argument ", name, data}]]
		];
		CreateMExpr @@{elem}
	]
	
getVariable[self_, id_] :=
	self["variables"]["lookup", id, Null]
	
setVariable[self_, id_, var_] :=
	self["variables"]["associateTo", id -> var]
	
getBasicBlock[self_, "BasicBlockID"[id_]] :=
	self["basicblocks"]["lookup", id, Null]
	
setBasicBlock[self_, id_, bb_] :=
	self["basicblocks"]["associateTo", id -> bb]

ClearAll[typeSameQ]

typeSameQ[ self_, t1_?TypeObjectQ, t2_?TypeObjectQ] :=
    t1["unresolve"] === t2["unresolve"]
typeSameQ[ self_, t1_?TypeObjectQ, t2_] :=
  typeSameQ[ self, t1, self["typeEnvironment"]["resolve", t2]]

typeSameQ[ self_, t1_, t2_?TypeObjectQ] :=
  typeSameQ[ self, self["typeEnvironment"]["resolve", t1], t2]

typeSameQ[ self_, (Type|TypeSpecifier)[t1_], (Type|TypeSpecifier)[t2_]] :=
    t1 === t2
 
typeSameQ[ self_, Undefined, Undefined ] :=
	True

typeSameQ[ self_, (Type|TypeSpecifier)[t1_], Undefined] := t1 === Undefined
typeSameQ[ self_, Undefined, (Type|TypeSpecifier)[t2_]] := t2 === Undefined


typeSameQ[args___] :=
	ThrowException[{"Unrecognized call to typeSameQ", {args}}]


RegisterCallback["DeclareCompileClass", Function[{st},
WIRSerializationEnvironmentClass = DeclareClass[
	WIRSerializationEnvironment,
	<|
		"getVariable" -> (getVariable[Self, #1]&),
		"setVariable" -> (setVariable[Self, #1, #2]&),
		"getBasicBlock" -> (getBasicBlock[Self, #1]&),
		"setBasicBlock" -> (setBasicBlock[Self, #1, #2]&),
		"getElement" -> (getElement[Self,#1,#2]&),
		"getElementList" -> (getElementList[Self,#1,#2]&),
		"getElementNoDeserialize" -> (getElementNoDeserialize[Self,#1,#2]&),
		"getElementMExpr" -> (getElementMExpr[Self,#1,#2]&),
		"serializeType" -> (serializeType[Self,#1]&),
		"deserializeType" -> (deserializeType[Self,#1]&),
		"getType" -> (deserializeType[Self, #1]&),
		"typeSameQ" -> (typeSameQ[Self,##]&),
		"deserialize" -> (deserialize[Self,#]&)
	|>,
	{
		"isClone" -> False,
		"uniqueID" -> False,
		"basicblocks",
		"variables",
		"typeEnvironment"
	},
	Predicate -> WIRSerializationEnvironmentQ
];
]]


CreateDeserializationEnvironment[tyEnv_, opts_:<||>] :=
	Module[{uniqueID},
		uniqueID = Lookup[ opts, "uniqueID", False];
		CreateObject[
			WIRSerializationEnvironment,
			<|
				"typeEnvironment" -> tyEnv,
				"uniqueID" -> TrueQ[uniqueID],
				"basicblocks" -> CreateReference[ <||>],
				"variables" -> CreateReference[ <||>]
			|>
		]
	]




deserializeClone[self_, obj_] :=
	obj["clone", self]


getElementClone[self_, obj_, name_] :=
	Module[ {elem},
		elem = obj[name];
		self["deserialize", elem]
	]


getElementListClone[self_, obj_, name_] :=
	Module[ {elems},
		elems = obj[name];
		If[!ListQ[elems],
			ThrowException[CompilerException[{"Argument is expected to be a list ", name, data}]]
		];
		Map[ self["deserialize",#]&, elems]
	]
	
getElementNoDeserializeClone[self_, obj_, name_] :=
	Module[ {elem},
		elem = obj[name];
		elem
	]
	

getElementMExprClone[self_, obj_, name_] :=
	Module[ {elem},
		elem = obj[name];
		elem["clone"]
	]


serializeType[self_, Undefined] := Undefined
serializeType[self_, ty_?TypeObjectQ] := ty["unresolve"]
serializeType[self_, ty:Type[t_]] := TypeSpecifier[t]
serializeType[self_, ty:TypeSpecifier[_]] := ty
serializeType[args___] := ThrowException[{"Bad arguments to serializeType: ", {args}}]

deserializeType[self_, ty_] :=
	If[ ty === Undefined,
		Undefined,
		self["typeEnvironment"]["resolve", ty]
	]

deserializeType[args___] :=
	ThrowException[{"Bad arguments to deserializeType: ", {args}}]

typeSameQClone[self_, serTy_, desTy_] :=
	serTy["unresolve"] === desTy["unresolve"]

typeSameQClone[args___] :=
	ThrowException[{"Unrecognized call to typeSameQClone", {args}}]

RegisterCallback["DeclareCompileClass", Function[{st},
CloneEnvironmentClass = DeclareClass[
	CloneEnvironment,
	<|
		"typeSameQ" -> (typeSameQClone[Self,##]&),
		"getElement" -> (getElementClone[Self,#1,#2]&),
		"getElementList" -> (getElementListClone[Self,#1,#2]&),
		"getElementNoDeserialize" -> (getElementNoDeserializeClone[Self,#1,#2]&),
		"getElementMExpr" -> (getElementMExprClone[Self,#1,#2]&),
		"deserialize" -> (deserializeClone[Self,#]&)
	|>,
	{
		"isClone" -> True
	},
	Extends -> {
		WIRSerializationEnvironmentClass
	},
	Predicate -> CloneEnvironmentQ
];
]]

CreateCloneEnvironment[opts_:<||>] :=
	Module[{uniqueID},
		uniqueID = Lookup[ opts, "uniqueID", True];
		CreateObject[
			CloneEnvironment,
			<|
				"uniqueID" -> TrueQ[uniqueID],
				"basicblocks" -> CreateReference[ <||>],
				"variables" -> CreateReference[ <||>]
			|>
		]
	]


GetSerializationData[ WIRSerialization[ "ProgramModule"[args_?AssociationQ], ___], name_] :=
	Lookup[args, name, <||>]
	
GetSerializationData[ args__] :=
	ThrowException[{"Unrecognized call to GetSerializationData", {args}}]
	
End[]


EndPackage[]
