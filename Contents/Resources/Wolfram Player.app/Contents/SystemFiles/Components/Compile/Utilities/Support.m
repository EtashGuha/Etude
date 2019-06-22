BeginPackage["Compile`Utilities`Support`"]


Begin["`Private`"]

Needs["Compile`"]
Needs["TypeFramework`"]
Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]


(*
 Defines the function Compile`Utilities`SupportedSymbols. This returns a list
 of symbols which are known to the Compilation system.
 
 Options:
    "TypeEnvironment",  the type environment to use,  default value $DefaultTypeEnvironment
    "OutputFilter",  a filter to apply to the results,  the default is to only 
    	return System context symbols
    	
  Currently this returns the TypePrediction symbols as System context,  maybe this should 
  have an option to filter these out since they require a Kernel.
*)


Options[Compile`Utilities`SupportedSymbols] =
	{
	"TypeEnvironment" -> Automatic,
	"OutputFilter" -> Automatic	
	}

Compile`Utilities`SupportedSymbols[ opts:OptionsPattern[]] :=
	Module[{tyEnv, macEnv, lowEnv, symList, filter},
		InitializeCompiler[];
		filter = getFilter[ OptionValue["OutputFilter"]];
		tyEnv = OptionValue["TypeEnvironment"];
		If[!TypeEnvironmentQ[env],
			tyEnv = $DefaultTypeEnvironment];
		macEnv = $DefaultMacroEnvironment;
		symList = Union[ 
					getTypeEnvironmentSymbols[tyEnv],
					getMacroSymbols[macEnv],
					getLoweringSymbols[ lowEnv]];
		symList = Select[ symList, filter];
		symList
	]

getFilter[Automatic] :=
	Function[{sym},
		Head[sym] === Symbol &&
			With[{cont = Context[sym]},
				cont === "System`"
			]]
			
		
getFilter[ filter_] :=
	filter

getTypeEnvironmentSymbols[tyEnv_] :=
	Join[
		tyEnv["functionTypeLookup"]["polymorphic"]["keys"],
		tyEnv["functionTypeLookup"]["monomorphic"]["keys"]]

getMacroSymbols[macEnv_] :=
	Module[{},
		macEnv["rules"]["keys"]
	]




getSymbol[ sym_String[][]] :=
	getSymbol[sym]
	
getSymbol[ sym_String[]] :=
	getSymbol[sym]

getSymbol[ sym_String] :=
	Module[{ef = Quiet[Symbol[sym]]},
		If[AtomQ[ef], ef, {}]
	]

getSymbol[ _] :=
	{}

getLoweringSymbols[lowEnv_] :=
	Module[{list},
		list = Keys[$LanguagePrimitiveLoweringRegistry];
		list = Map[ getSymbol, list];
		Flatten[list]
	]



End[]


EndPackage[]
