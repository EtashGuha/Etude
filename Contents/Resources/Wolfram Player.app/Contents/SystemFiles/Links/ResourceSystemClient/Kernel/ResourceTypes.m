(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}


BeginPackage["ResourceSystemClient`"]

Begin["`Private`"] (* Begin Private Context *) 

ClearAll[loadResourceType];

loadResourceType[rtype_String]:=Block[{pacletname=pacletName[rtype], contextname=contextName[rtype], res},
	res=checkPaclet[pacletname];
	If[res===$Failed,
		$Failed;
		loadResourceType[rtype]=$Failed;
		,
		Needs[contextname];
		addToResourceTypes[rtype];
		loadResourceType[rtype]=Null;
	]
]

pacletName["DataResource"|"Data"|"data"]="DataResource";
contextName["DataResource"|"Data"|"data"]="DataResource`";

pacletName[str_String]:=str<>"Resource"/;StringFreeQ[str,"Resource"]
contextName[str_String]:=str<>"Resource`"/;StringFreeQ[str,"Resource"]

pacletName[rtype_]:=rtype;
contextName[rtype_]:=rtype<>"`";

checkPaclet[pacletname_]:=Block[{paclets},
	paclets=PacletManager`PacletFind[pacletname];
	If[Length[paclets]>0,
		Return[]];
	paclets=PacletManager`PacletFindRemote[pacletname];
	If[Length[paclets]>0,
		PacletManager`PacletInstall[pacletname]
		,
		Message[ResourceObject::unkrt,pacletname];
		$Failed
		]
]

$availableResourceTypes={None};
addToResourceTypes[rtype_]:=($availableResourceTypes=DeleteDuplicates[Flatten[{$availableResourceTypes,rtype}]])



(******************************************************************************)

setAutoLoad // ClearAll;
setAutoLoad // Attributes = { HoldRest };

setAutoLoad[ type_, symbol_Symbol ] :=
  PreemptProtect @ Once @ Internal`WithLocalSettings[
      Unprotect @ symbol;
      ClearAll @ symbol;
      ,
      symbol :=
        Module[ { loaded, success },

            loaded = Quiet[ Check[ loadResourceType @ type,
                                   $Failed,
                                   ResourceObject::unkrt
                            ],
                            ResourceObject::unkrt
                     ];

            success = TrueQ @ ! FailureQ @ loaded;

            If[ ! success, messageMissingPaclet[ symbol, type ] ];

            symbol /; success
        ]
      ,
      SetAttributes[ symbol, { ReadProtected, Protected } ]
  ];


messageMissingPaclet // ClearAll;
messageMissingPaclet // Attributes = { HoldFirst };

messageMissingPaclet[System`ResourceFunction, "Function"]:=
	messageMissingPaclet[System`ResourceFunction, "Function"]=
		Message[ ResourceFunction::frpaclet ]
messageMissingPaclet[ symbol_Symbol, type_ ] :=
  messageMissingPaclet[ symbol, type ] =
    Message[ symbol::respaclet, type ];


(******************************************************************************)
(* Symbols that depend on loading a resource type                             *)
(******************************************************************************)

setAutoLoad[ "Function", System`ResourceFunction ];


(******************************************************************************)



End[] (* End Private Context *)

EndPackage[]

SetAttributes[{},
   {ReadProtected, Protected}
];
