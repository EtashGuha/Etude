


Begin["Chemistry`Private`OPSINLinkDump`"]


Needs["JLink`"]

NameToSMILES::usage = "NameToSMILES[chem] parses the chemical name chem to a SMILES string using the OPSIN parser." 



(* ************************************************************************* **

                        NameToSMILES

** ************************************************************************* *)



Options[ NameToSMILES ] = {
	"AllowRadicals" -> True, "AddDummyAtoms" -> False,
	"FailOnBadStereochemistry" -> True , "Canonicalize" -> True
}

NameToSMILES[name: _String , opts:OptionsPattern[]] :=
    Block[{res},
        res = nameToSMILES[name, opts];
        res /; MatchQ[res, _String | _Failure]
    ]



(* :nameToSMILES: *)

$NameToStructure;
$ntsConfig;

getOPSINinstance[] := If[
	AllTrue[{$NameToStructure,$ntsConfig}, JavaObjectQ]
	,
	{$NameToStructure,$ntsConfig}
	,
	InstallJava[];
	LoadJavaClass["uk.ac.cam.ch.wwmm.opsin.NameToStructure", AllowShortContext -> False];
	LoadJavaClass["uk.ac.cam.ch.wwmm.opsin.NameToStructureConfig", AllowShortContext -> False];
	LoadJavaClass["java.lang.Enum", AllowShortContext -> False];
	LoadJavaClass["uk.ac.cam.ch.wwmm.opsin.OpsinResult$OPSIN_RESULT_STATUS", AllowShortContext -> False];
	Clear[$NameToStructure];
	Clear[$ntsConfig];
	$ntsConfig = JavaNew @ "uk.ac.cam.ch.wwmm.opsin.NameToStructureConfig";
	$NameToStructure = uk`ac`cam`ch`wwmm`opsin`NameToStructure`getInstance[];
	{$NameToStructure,$ntsConfig}
]

getOpsinResult[ opsin_, chem_] := Switch[opsin @ getStatus[][name[]],
	"SUCCESS",
		opsin @ getSmiles[],
	"WARNING",
		issueOpsinMessage[opsin, chem];
		opsin @ getSmiles[],
	"FAILURE",
		Failure[
			"OpsinParseFailure",
			getOpsinFailureAssociation[opsin, chem]
		]
]

nameToSMILES[args___] := Block[{javaLoaded},
	ClearAll @ nameToSMILES;
	If[
		MatchQ[InstallJava[], $Failed]
		,
		nameToSMILES[___] := Failure["OpsinNotLoaded", <|"Message" -> "OPSIN parser not loaded."|>]
		,
		nameToSMILES[name_String, opts:OptionsPattern[]] := Module[
			{nts,ntsc,opsinResult,allowRadicals, addDummyAtoms, failOnBadStereochemistry, canonicalize},
			{nts,ntsc} = getOPSINinstance[];
			{allowRadicals, addDummyAtoms, failOnBadStereochemistry, canonicalize} = System`Utilities`GetOptionValues[
				NameToSMILES,
				Part[Options @ NameToSMILES, All, 1],
				{opts}
			];
			canonicalize = If[ TrueQ @ canonicalize, SmilesToCanonicalSmiles, Identity];
			
			(*
				This checks that the options for the opsin parser match the current options,
				and if not then it calls the java option setter.
			*)
			If[
				ntsc @ #1[] =!= #2
				,
				ntsc @ #3[#2]
			] & @@@ {
				{isAllowRadicals, allowRadicals, setAllowRadicals},
				{isOutputRadicalsAsWildCardAtoms, addDummyAtoms, setOutputRadicalsAsWildCardAtoms},
				{warnRatherThanFailOnUninterpretableStereochemistry, failOnBadStereochemistry, 
					setWarnRatherThanFailOnUninterpretableStereochemistry}
			};
			
			JavaBlock[
				opsinResult = nts @ parseChemicalName[name, ntsc];
				canonicalize @ getOpsinResult[opsinResult, name]
			] /; AllTrue[{nts,ntsc}, JavaObjectQ]
		];
		
		nameToSMILES[___] := $Failed;
	];
	nameToSMILES @ args
]  		

issueOpsinMessage[opsinResult_,inputName_] :=  Module[{warning,type,message},
	warning = First @ JavaObjectToExpression @ opsinResult@getWarnings[];
	type = warning@getType[]@name[];
	message = fixOpsinMessage[warning@getMessage[]];
	If[ StringContainsQ[ message, "OPSIN"] || TrueQ[SuppressOpsinMessage], Return[Null, Module]];
	Switch[
		type,
		"STEREOCHEMISTRY_IGNORED",
		Message[Molecule::stereo2, inputName, message],
		"APPEARS_AMBIGUOUS",
		Message[Molecule::ambig, inputName, message]
	];
];

 


fixOpsinMessage[s_String] := StringReplace[s,
	{
		RuleDelayed[
			StringExpression[bef:("Could not find"~~Shortest[__])~~": <",
				__, ">", val__, "</" ~~ __ ~~ ">"~~__~~"referring to"
			],
			StringJoin[bef," ", val, " refers to."]
		]
	}
] 


$OpsinFailureString1 = nam__ ~~ "is unparsable due to the following being uninterpretable: " ~~ uint__ ~~ " The following was not parseable: " ~~ pars__;



getOpsinFailureAssociation[ opsinResult_, name_] := Module[
	{opsinMessage = opsinResult@getMessage[]},
	First @ StringCases[
		opsinMessage,
		{
			RuleDelayed[
				$OpsinFailureString1,
				<|"Input" -> nam, "Uninterpretable" -> uint, "Unparseable" -> pars|>
			],
			x__ :> <|"Input" -> name, "Message" -> x|>
		}
	] 
]


End[] (* End Private Context *)


