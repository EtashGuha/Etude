
Needs["CompileUtilities`Asserter`Common`"]

With[{$FailureSymbol = $FailureSymbol},

failureString[$FailureSymbol[tag_, meta_Association /; KeyExistsQ[meta, "MessageTemplate"]]] :=
        FormatMessage[meta["MessageTemplate"], Lookup[meta, "MessageParameters", {}]];
failureString[$FailureSymbol[tag_String, ___]] := "A failure of type \"" <> tag <> "\" occurred.";
failureString[_$FailureSymbol] := "An unknown failure occurred.";

(f:$FailureSymbol[tag_, meta_Association])["Message"] := failureString[f];
(f:$FailureSymbol[tag_, meta_Association])["StyledMessage"] := Style[failureString[f], "Message"];
makeGrid[assoc_] := BoxForm`SummaryItem[{ToString[#1] <> ": ", #2}]& @@@ Normal[assoc];

$FailureSymbol /: MakeBoxes[f:$FailureSymbol[tag_, messagetemp_, args_:<||>], fmt_] :=
	BoxForm`ArrangeSummaryBox[
	        $FailureSymbol,
	        f,
	        Style["\[WarningSign]", Directive["Message", 35]],
	        Join[
	        	If[KeyExistsQ[args, "Description"],
	        		{
	        			BoxForm`SummaryItem[{"Description: ", args["Description"]}]
	        		},
	        		{}
	        	],
	        	{
		                BoxForm`SummaryItem[{"Message: ", failureString[f]}],
		                BoxForm`SummaryItem[{"Tag: ", tag}]
		        }
	        ],
	        makeGrid[KeyDropFrom[args, "Description"]],
	        fmt
	];
$FailureSymbol /: MakeBoxes[f:$FailureSymbol[tag_, meta_Association /; KeyExistsQ[meta, "MessageTemplate"] ], fmt_] :=
	BoxForm`ArrangeSummaryBox[
	        $FailureSymbol,
	        f,
	        Style["\[WarningSign]", Directive["Message", 35]],
	        Join[
	        	If[KeyExistsQ[meta, "Description"],
	        		{
	        			BoxForm`SummaryItem[{"Description: ", meta["Description"]}]
	        		},
	        		{}
	        	],
	        	{
		                BoxForm`SummaryItem[{"Message: ", failureString[f]}],
		                BoxForm`SummaryItem[{"Tag: ", tag}]
		        }
	        ],
	        makeGrid[Delete[meta, {{"MessageTemplate"}, {"MessageParameters"}, {"Description"}}]],
	        fmt
	];
]

Format[f:$FailureSymbol[tag_, messagetemp_, args_:<||>], OutputForm] :=
(
	ToString[$FailureSymbol] <> "[" <>
	If[KeyExistsQ[args, "Description"], "Description: " <> args["Description"], ""] <> " " <>
	"Message: " <> failureString[f] <> " " <>
	"Tag: " <> tag <> " " <>
	ToString[KeyDropFrom[args, "Description"]] <>
	"]"
)
Format[f:$FailureSymbol[tag_, meta_Association /; KeyExistsQ[meta, "MessageTemplate"]], OutputForm] :=
(
	ToString[$FailureSymbol] <> "[" <>
	If[KeyExistsQ[meta, "Description"], "Description: " <> meta["Description"], ""] <> " " <>
	"Message: " <> failureString[f] <> " " <>
	"Tag: " <> tag <> " " <>
	ToString[Delete[meta, {{"MessageTemplate"}, {"MessageParameters"}, {"Description"}}]] <>
	"]"
)

Format[f:$FailureSymbol[tag_, messagetemp_, args_:<||>], TextForm] :=
(
	ToString[$FailureSymbol] <> "[" <>
	If[KeyExistsQ[args, "Description"], "Description: " <> args["Description"], ""] <> " " <>
	"Message: " <> failureString[f] <> " " <>
	"Tag: " <> tag <> " " <>
	ToString[KeyDropFrom[args, "Description"]] <>
	"]"
)
Format[f:$FailureSymbol[tag_, meta_Association /; KeyExistsQ[meta, "MessageTemplate"]], TextForm] :=
(
	ToString[$FailureSymbol] <> "[" <>
	If[KeyExistsQ[meta, "Description"], "Description: " <> meta["Description"], ""] <> " " <>
	"Message: " <> failureString[f] <> " " <>
	"Tag: " <> tag <> " " <>
	ToString[Delete[meta, {{"MessageTemplate"}, {"MessageParameters"}, {"Description"}}]] <>
	"]"
)

Format[f:$FailureSymbol[tag_, messagetemp_, args_:<||>], ScriptForm] :=
(
	ToString[$FailureSymbol] <> "[" <>
	If[KeyExistsQ[args, "Description"], "Description: " <> args["Description"], ""] <> " " <>
	"Message: " <> failureString[f] <> " " <>
	"Tag: " <> tag <> " " <>
	ToString[KeyDropFrom[args, "Description"]] <>
	"]"
)
Format[f:$FailureSymbol[tag_, meta_Association /; KeyExistsQ[meta, "MessageTemplate"]], ScriptForm] :=
(
	ToString[$FailureSymbol] <> "[" <>
	If[KeyExistsQ[meta, "Description"], "Description: " <> meta["Description"], ""] <> " " <>
	"Message: " <> failureString[f] <> " " <>
	"Tag: " <> tag <> " " <>
	ToString[Delete[meta, {{"MessageTemplate"}, {"MessageParameters"}, {"Description"}}]] <>
	"]"
)
