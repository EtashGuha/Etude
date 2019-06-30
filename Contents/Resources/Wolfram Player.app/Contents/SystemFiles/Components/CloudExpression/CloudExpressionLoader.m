(* All loading of the paclet's Wolfram Language code should go through this file. *)

(* Developer maintains this list of symbols.
   autoloadSymbols must agree with the symbols listed in the Kernel extension in the PacletInfo.m file.
*)        

PacletManager`Package`loadWolframLanguageCode["CloudExpression", "CloudExpression`", DirectoryName[$InputFileName], 
	"Main.m",
	"AutoUpdate" -> False, "ForceMX" -> False, "Lock" -> False, 
	"HiddenImports" -> {"Macros`", "GeneralUtilities`"},
	"AutoloadSymbols" -> {
		"System`CloudExpression",
		"System`CreateCloudExpression",
		"System`DeleteCloudExpression",
		"System`CloudExpressions",
		"System`PartProtection",
		"System`$CloudExpressionBase"
	}
];

CloudObject;

Unprotect[System`$CloudExpressionBase];

Language`SetMutationHandler[CloudExpression, CloudExpression`PackageScope`CloudExpressionHandler];

General::invrsp = "An invalid server response was received.";
General::cemiss = "The cloud expression does not exist or you are not permitted to access it."
General::ceopfail = "Cloud expression operation failed: ``";
General::cenotperm = "Cloud expression operation not permitted: ``";
General::ceputpart = "Put is not applicable to the parts of a cloud expression."
General::invceb = "Invalid value for $CloudExpressionBase.";
General::invce = "Invalid CloudExpression ``";
