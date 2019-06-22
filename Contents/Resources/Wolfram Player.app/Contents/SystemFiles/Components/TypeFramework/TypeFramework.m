(* Wolfram Language Package *)

(* Created by the Wolfram Workbench 01-Jun-2017 *)

BeginPackage["TypeFramework`"]
(* Exported symbols added here with SymbolName::usage *)

TypeEnvironment
TypeEnvironmentQ
TypeObjectQ
AbstractTypeQ
TypeResolvableQ
TypeInferenceException


TypeError

AbstractType
TypeConstructor
TypeInstance
TypeVariable
TypeForAll
TypePredicate
TypeQualified
TypeAssumption
TypeSequence
TypeApplication
TypeRecurse
TypeLiteral
TypeAlias
TypeProjection
TypeEvaluate

MetaData
FunctionData

AlternativeConstraint
AssumeConstraint
EqualConstraint
FailureConstraint
GeneralizeConstraint
LookupConstraint
InstantiateConstraint
ProveConstraint
SkolemConstraint
SuccessConstraint



ConstraintSolveForm

InitializeTypeFrameworkClasses

Begin["`Private`"]

Needs["TypeFramework`TypeObjects`TypeBase`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Format`"]
Needs["TypeFramework`Utilities`Error`"]

Needs["TypeFramework`ConstraintObjects`"];
Needs["TypeFramework`Environments`"];
Needs["TypeFramework`Inference`"];
Needs["TypeFramework`TypeObjects`"];



InitializeTypeFrameworkClasses[] := InitializeTypeFrameworkClasses[] = (
	SortCallbacks["DeclareTypeFrameworkClass", SortClassesFunction];
	RunCallback["DeclareTypeFrameworkClass", {}];
)

TypeObjectQ = TypeBaseClassQ



TypeResolvableQ[env_?TypeEnvironmentQ, ty_?TypeObjectQ] :=
	True

TypeResolvableQ[env_?TypeEnvironmentQ, ty_?AbstractTypeQ] :=
	True

TypeResolvableQ[env_?TypeEnvironmentQ, ty_] :=
CatchTypeFailure[
	ef = env["resolve", ty];
	TypeObjectQ[ef] || AbstractTypeQ[ef]
	,
	_
	,
	False&
]

TypeResolvableQ[args___] :=
	TypeFailure["TypeResolve", "Unrecognized call to TypeResolvableQ", args]







End[]

EndPackage[]

