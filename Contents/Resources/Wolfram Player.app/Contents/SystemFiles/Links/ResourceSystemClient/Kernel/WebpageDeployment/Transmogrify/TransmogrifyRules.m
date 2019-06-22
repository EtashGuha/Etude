
ResourceShingleTransmogrify`Private`resourcesystemXMLrules;
BeginPackage["ResourceShingleTransmogrify`"]
Begin["`Private`"]

resourcesystemXMLrules=XMLTransform[{

_Notebook :>  DIV[{"class" -> "example-notebook"}, {Recurse[] }],

{Cell, "DataResourceExamplesChapter"} :> "",
{Cell, "Chapter"} :> "",

{Cell, "Section"} :> DIV[{"class" -> "example-section"}, {Recurse[] }],

{Cell, "Subsection"} :> H3[{"class" -> "example-subsection"}, {Recurse[] }],

{Cell, "Subsubsection"} :> H4[{"class" -> "example-subsubsection"}, {Recurse[] }],

{Cell, "Text"} :> P[{"class" -> "example-text"}, {Recurse[] }],

{CellGroupData, "Input"} :> DIV[ {"class"->"example-frame"}, {Recurse[] }],

{Cell, "Input"} :>  
Module[{inputForm,self=SelectSelf[], inlabel},
	inputForm = CreateInputForm[ self ];
	If[inputForm===None,
		Return[Null]
	];
	inputForm = If[Head@inputForm === String, inputForm, ToString@inputForm];
	altAttribute = StringReplace[inputForm, "\"" -> "&quot;"];
	
	inlabel=getInputLabel[self];
	
	XMLElement["table", {"class" -> "example input"}, {
		XMLElement["tr", {}, {
			TD[{"class" -> "in-out"}, {
				"In[" <> inlabel <> "]:="
			}],
			TD[{}, { DIV[ {"class" -> "img-frame"}, { 
				Image[FileNameJoin[{GetParameter["FileName"] <> "-io-" <> inlabel <> "-i" <> GetParameter["langext"] <> ".gif"}],
					{"alt" -> altAttribute, "data-alt-length" -> StringLength[altAttribute] },
					Inline -> True, CropImage -> False
				]
			}] }]
		}]
	}]
],


{Cell, "Output"} :>  
Module[{outlabel},
	
	outlabel=getOutputLabel[SelectSelf[]];
	
	XMLElement["table", {"class" -> "example output"}, {
		XMLElement["tr", {}, {
			TD[{"class" -> "in-out"}, {
				"Out[" <> outlabel <> "]="
			}],
			TD[{}, {
				Image[FileNameJoin[{GetParameter["FileName"] <> "-io-" <> outlabel <> "-o" <> GetParameter["langext"] <> ".gif"}],
					{"class" -> "output", "alt" -> "" },
					Inline -> True, CropImage -> False
				]
			}]
		}]
	}]
],



(****************************)
(*	Miscellaneous Heads 	*)
(****************************)

{InterpretationBox} :> Recurse[],

{BoxData}:> Recurse[],

{SuperscriptBox} :> Sequence@@{
	ResourceShingleTransmogrify`ResourceShingleTransmogrify[SelectChildren[][[1]] ],
	XMLElement["sup", {} , {ResourceShingleTransmogrify`ResourceShingleTransmogrify[SelectChildren[][[2]]]} ]
},

{SubscriptBox} :> Sequence@@{
	ResourceShingleTransmogrify`ResourceShingleTransmogrify[SelectChildren[][[1]] ],
	XMLElement["sub", {}, {ResourceShingleTransmogrify`ResourceShingleTransmogrify[SelectChildren[][[2]]]} ]
},

{String} :> 
Module[{stringContent, stringTest}, 
	
	stringContent = SelectSelf[];
	stringTest[x_] := 
	Which[
		Head[x] === XMLElement, x,
		Head[x] === RawXML, x,
		StringLength[x] === 1 && StringMatchQ[x, RegularExpression["[a-z]"]], XMLElement["I", {}, {x}], 
  		True, x
  	];
	
	(* handle inline formatted strings... *)
	Which[
		InlineFormattedQ[SelectLiteral[SelectSelf[]]],
  			ResourceShingleTransmogrify`ResourceShingleTransmogrify[FromInlineString[SelectSelf[]]],
    	traditionalFormQ === True, 
			Sequence@@Map[stringTest[#]&,  {stringContent}],
		True, stringContent
	]
],



(****************************************)
(*	COMMON STYLES but possibly not used	*)
(****************************************)

{Cell, "Print"} :>
	DIV[{"class" -> "Print"}, {
		Image[FileNameJoin[{ 
			GetParameter["FileName"] <> "-image-" <> IncrementCounter["Image"] <> GetParameter["langext"] <> ".gif"}],
			{"alt" -> ""},
			Inline -> False, 
			CropImage -> False]
	}],

{Cell, "Graphics"} :>
	DIV[{"class" -> "Graphics"}, {
		Image[FileNameJoin[{ 
				GetParameter["FileName"] <> "-a-" <> IncrementCounter["Animation"] <> GetParameter["langext"] <> ".gif"}], 
			{"alt" -> ""},
			Inline -> False ]
	}],

{Cell, "Picture"} :>
	DIV[{"class" -> "Picture"}, {
		Image[FileNameJoin[{
				GetParameter["FileName"] <> "-pict-" <> IncrementCounter["Picture"] <> GetParameter["langext"] <> ".gif"}], 
			{"alt" -> ""},
			Inline -> True]
	}], 


(* Styleless Cells *)
{Cell, None} :> 
	If[htmlBoxesQ[ SelectLiteral[ SelectSelf[] ] ], 
 		Recurse[]
 	, 
		Image[FileNameJoin[{ 
				GetParameter["FileName"] <> "-box-" <> IncrementCounter["Inline"] <> GetParameter["langext"] <> ".gif"}],
			{alt -> ""}, 
			Inline -> True, 
			CropImage -> False]
     ],


{GraphicsBox | PanelBox | PaneBox | Graphics3DBox | OverlayBox} :> 
	Image[FileNameJoin[{ 
			GetParameter["FileName"] <> "-" <> IncrementCounter["Image"] <> GetParameter["langext"] <> ".gif"}],
		{"alt" -> ""},
		Inline -> True, 
		CropImage -> False
]



},

DefaultParameters -> {

"noop" -> {

(* CreateInputForm text form for copyable popups *)
CreateInputForm::str = "Incorrect output: `1`";

CreateInputForm[HoldPattern[Cell["",___]], ___]:=None;

CreateInputForm[ce : Cell[___], opts___] := 
  Developer`UseFrontEnd@Module[{nbExpr, expr, c = ce},
    If[IsManipulateQ[c], " ",
		If[IsLinguisticAssistantQ[c], 
			(* free-form input *)
			c = c /. nsb:NamespaceBox[__] :> NamespaceBoxToString[nsb];
		]; 
		(
	      nbExpr = MathLink`CallFrontEnd[ExportPacket[c, "InputText"]];
	      expr = GetFirstCellContents[nbExpr];
	      expr = If[StringQ@expr, 
	        expr = ConvertToString@expr;
	        expr = StringReplace[expr, "\\[ThinSpace]" -> ""];
	        FixedPoint[StringReplace[#, RegularExpression["(\r|\n)+"] :> "\n"]&, expr],
	        (* Else *)
	        expr];
	      If[(Head@expr =!= String),
	        If[!FAILED,
	          FAILED=True;
	          Export["CreateInputFormFailed.m", {Global`PageTitleForCreateInputForm, c, nbExpr, expr}];
	        ];
	        Message[CreateInputForm::str,expr]; $Failed, 
	        expr] 
		)
	]
];

(**
    inputQueue[] - get the appropriate list of Inputs up to and including an
    inputline containing a "%".

    Expects "s" to be the result of CreateInputForm[]. Also, use RenumberCells[]
    to make sure each example section starts at In[1] for best results. 
*)
inputQueue[s_String,label_] := Module[{out, refs},
  Catch[
    (* reset the queue if we've started a new section.
       we're assuming that the first input cell does not contain a
       "%", as that would just be silly... :) *)
    If[!StringFreeQ[label/.None->"","In[1]"],
      ClearAll[$InputQueue];
      $InputQueue[$InputCount=1]=s;
      Throw[s]
    ];
                                                                                               
    (* now add to the queue ...
       note that $IntegerCount _should_ already be initialized above *)
    $InputQueue[++$InputCount] = s;

    Which[StringFreeQ[s,"%"],
      (* return the string if there's no "%" or ... *)
      s,
      !StringFreeQ[$InputQueue[1],"%"],
      s,
      True,
      (* ... otherwise, return the specified input requested by "%". *)
      (* join all the % references in the queue, separated by ";\n" *)
      (StringJoin@@((#<>";\n"&)/@Most[#])<>Last[#])&[findInputRefs[$InputQueue[$InputCount],{$InputCount}]]
    ]
  ]
];
inputQueue[f___] := "$Failed";


(*  fun recursion to get all %-referenced inputs.
    i'm sure there's a non-Recursive-MapIndexed way to do this
    but this seemed easiest. :) *)
findInputRefs[s_String,p_]:= Module[{spl, pos = p[[1]], refs},
  spl = percentSplit[s];
  (* FIXME: this doesn't handle %1 digitized refs, although
     percentSplit does.  I don't think we use those in our
     documentation though.
  *)
  (* FIXME: this is actually broken when refs contains % refs,
     as it gets them all out of order and doesn't uniqueify
     them in any meaningful way.  Try this:
       $InputQueue[++$InputCount] = "One[1]";
       $InputQueue[++$InputCount] = "Two[%]";
       $InputQueue[++$InputCount] = "Three[%]";
       $InputQueue[++$InputCount] = "Four[%, %%]";
       findInputRefs[$InputQueue[$InputCount], {$InputCount}]
     This bug has existed for a while, but I don't think we
     actually trigger it anywhere.
  *)
  refs = Flatten[
    If[# > 0 && pos =!= #, $InputQueue[pos - #], $InputQueue[pos]] & /@ 
      Reverse[Sort[StringLength /@
        UnsortedUnion[Cases[spl, pc_/;(StringTake[pc, 1] === "%")]]
    ]] ];
  Flatten@Append[MapIndexed[findInputRefs, refs], s]
];


(**
  htmlBoxesQ
  check to see if *Box's need to be images 
**)

htmlBoxesQ[expr_]:=
	Module[{}, 
	Length[Select[
		Position[expr,
			(_FractionBox | _RadicalBox | _SqrtBox | _OverscriptBox | _UnderscriptBox | _SubsuperscriptBox | _GridBox | _SliderBox | _Slider2DBox | _CheckboxBox | _RadioButtonBox | _OpenerBox | _ActionMenuBox | _PopupMenuBox | _ProgressIndicatorBox | _InputFieldBox | _ColorSetterBox ),
			{1, Infinity}],
		Function[{pos},
			FreeQ[
			Map[Head[Extract[expr, #]] &,
				Map[Take[pos, #] &, Range[2, Length[pos]]]
			],
			Cell
		]]
	]]=== 0];


(** IsManipulateQ **)
IsManipulateQ[___]:= False;
IsManipulateQ[c_Cell]:= Not[FreeQ[c, Manipulate`InterpretManipulate]];

(** IsLinguisticAssistantQ **)
IsLinguisticAssistantQ[___]:= False;
IsLinguisticAssistantQ[c_Cell]:= Not[FreeQ[c, "LinguisticAssistant"]];

NamespaceBoxToString[ce_NamespaceBox]:=
Module[{res},
	(* Find free-form query *)
	res = Cases[ce, DynamicModuleBox[{___, Set[Typeset`query$$, query_], ___}, ___] :> query, Infinity];
	Cell[TextData[{StringJoin["(", ConvertToString[ RowBox[{ res }]], ")"] }]]
];
NamespaceBoxToString[a___]:= a;

(** IsSoundQ **)
IsSoundQ[___]:= False;
IsSoundQ[c_Cell]:= Not[FreeQ[c, Sound]];

(* get string form of first cell in nb *)
GetFirstCellContents[nb_Notebook] := 
  GetFirstCellContents[Cases[nb, Cell[c_, ___] :> c, Infinity]];
GetFirstCellContents[{l_, ___}] := GetFirstCellContents[l];
GetFirstCellContents[s_String] := s;
GetFirstCellContents[___] := $Failed;

}

}
]

End[]

EndPackage[]
ResourceShingleTransmogrify`Private`resourcesystemXMLrules