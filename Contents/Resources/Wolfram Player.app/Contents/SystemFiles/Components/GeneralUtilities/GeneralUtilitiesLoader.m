(* All loading of the paclet's Wolfram Language code should go through this file. *)

(* Developer maintains this list of symbols.
   autoloadSymbols must agree with the symbols listed in the Kernel extension in the PacletInfo.m file.
*)

GeneralUtilities`Private`autoloadSymbols = {
            "System`DeleteMissing",
            "System`DisjointQ", 
            "System`IntersectingQ", 
            "System`SubsetQ",
            "System`Failure",
            "System`PowerRange",
            "System`CountDistinct",
            "System`CountDistinctBy",
            "System`DeleteDuplicatesBy",
            "System`TextString",
            "System`AssociationMap",
            "System`InsertLinebreaks",
			"System`StringPadLeft",
			"System`StringPadRight",
			"System`StringExtract",
			"System`Decapitalize",
			"System`StringRepeat",
			"System`StringRiffle",
			"System`PrintableASCIIQ",  
			"System`AssociationFormat","System`ListFormat",
			"System`BooleanStrings", "System`MissingString",
			"System`TimeFormat", "System`ElidedForms",

(*			"System`Softmax",
			"System`MeanSquaredDistance",
			"System`MeanAbsoluteDistance",
			"System`CrossEntropy",
*)

			"GeneralUtilities`MLExport",
			"GeneralUtilities`MLImport",
			"GeneralUtilities`ValidPropertyQ",
			"GeneralUtilities`InformationPanel",
			"GeneralUtilities`ComputeWithProgress",
			"GeneralUtilities`TightLabeled"
}

General::notassoc = "First argument `` is not a symbol whose value is an association.";
General::nomon = "Progress of function `` cannot be automatically monitored.";

General::noreturnscope = "There is no valid outermost head to Return to after macro expansion: ``.";
General::badreturnscope = "The outermost head `` in `` is not a valid target for a Return.";
General::nomessagehead = "There is no head with which to associate the message in ``.";
General::invmacro = "Encountered invalid invocation of `` macro during expansion of ``.";
General::holdmacro = "`` macro cannot be used with symbol ``, which has Hold-type attributes.";
General::unmatched = "The case `` was unmatched by ``.";

General::interr2 = "An unknown internal error occurred. Consult Internal`.`$LastInternalFailure for potential information.";

General::invsrcarg = "Argument `` is not a paclet name, file, or directory.";
General::invsrcdir = "Could not find an entry point within the directory ``.";
General::invsrcpac = "Could not find a location on disk for the context ``.";
General::invcont = "`` is not a valid context or list of contexts."
General::invsrcfile = "Cannot load `` in isolation, because it is a new-style package fragment."
General::nosyms = "No symbols match ``."

General::invspatt = "The argument `1` is not a valid string pattern."
General::lists = "List of lists expected at position `2` in `1`."
General::stringnz = "String of non-zero length expected at position `2` in `1`."
General::strlist = "List of strings expected at position `2` in `1`."

General::asrtf = "Assertion `1` failed.";

General::elmntav = "`2` is not an available `1`.";
General::elmntavs = "`2` is not an available `1`. Did you mean `3` instead?";
General::elmntavl = "`2` is not an available `1`. Possible `3` include `4`.";
General::elmntavsl = "`2` is not an available `1`. Did you mean `3` instead? Possible `4` also include `5`.";

General::invmethparam = "Value of option Method -> `` specifies the invalid parameter setting ``."
General::invmethparam2 = "Value of option Method -> `` specifies the invalid parameter setting ``, which should be ``."
General::invmethodspec = "Value of Method option should be a string or a string with options."
General::invmethname = "Method name specified in option Method -> `` is not ``.";
General::unkmethparam = "Value of option Method -> `` specifies an unknown parameter ``. The parameters that are valid for method \"``\" are ``.";
General::nomethparams = "Value of option Method -> `` specifies parameters, but method \"``\" allows no parameters."

PreemptProtect[ (* <- prevent FE from loading some other paclet in the middle of our loading *)
	Block[{GeneralUtilities`$MacroDebugMode = False},
		Quiet[
		PacletManager`Package`loadWolframLanguageCode["GeneralUtilities", "GeneralUtilities`", DirectoryName[$InputFileName], "Code.m",
					"AutoUpdate" -> False, "ForceMX" -> False, "Lock" -> False, 
					"AutoloadSymbols" -> GeneralUtilities`Private`autoloadSymbols
		],
		{General::shdw, RuleDelayed::rhs}
		];
	];
]

