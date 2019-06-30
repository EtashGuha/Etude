
BeginPackage["CompileUtilities`Markup`"]


GrayText
VeryLightGrayText
BoldGrayText
BoldRedText
BoldBlackText
BoldGreenText
InstructionNumberText
InstructionNameText
LabelText
PropertyText


$UseANSI
$UseHTML

Begin["`Private`"]


(*

Markup strings with color, bold, and underline.

Uses StyleBox for displaying in the FrontEnd.

Uses ANSI escape codes for displaying in the console.


The flag $UseANSI can be set to False to disable ANSI sequences printed to console

*)


$UseANSI = False
$UseHTML = False
$UseHTML5 = False


colorANSICode[GrayLevel[gray_]] :=
	With[{code = ToString[232 + Round[gray*23]]},
		"\[RawEscape][38;5;"<>code<>"m"
	]

colorANSICode[RGBColor[r_, g_, b_]] :=
	With[{code = ToString[16 + 36*Round[5*r] + 6*Round[5*g] + Round[5*b]]},
		"\[RawEscape][38;5;"<>code<>"m"
	]

    
colorHTML[c:(_GrayLevel | _RGBColor)] :=
    "color:" <> ColorToHTMLColor[c]

weightANSICode[Bold] = "\[RawEscape][1m"
weightANSICode[_] = ""

weightHTML[Bold] = "font-weight:bold"
weightHTML[_] = ""

variationsANSICode[{"Underline"->True}] = "\[RawEscape][4m"
variationsANSICode[_] = ""

variationsHTML[{"Underline"->True}] = "text-decoration:underline"
variationsHTML[_] = ""

resetANSICode[] = "\[RawEscape][0m"

(*   ColorToHTMLColor  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
$defaultColor = RGBColor[0, 0, 0];
colorToHTMLColor[color:_Hue | _GrayLevel | _CMYKColor] := colorToHTMLColor[ToColor[color, RGBColor]];
colorToHTMLColor[x_RGBColor] := "#"<>StringJoin@@
Map[ToString,
Flatten@Map[IntegerDigits[IntegerPart[#],16,2]&,255*List@@x]/.{10-> "A",11-> "B",12-> "C",13-> "D",14-> "E",15-> "F"}
];
colorToHTMLColor[other_] := colorToHTMLColor[$defaultColor];



Options[markup] = {FontColor->White, FontWeight->Plain, FontVariations->{}}

markup[args___List, OptionsPattern[]] :=
	Module[{s, color, weight, variations},
		
		s = StringJoin[args];
		
		color = OptionValue[FontColor];
		weight = OptionValue[FontWeight];
		variations = OptionValue[FontVariations];

		Which[
			$FrontEnd =!= Null,
			"\!\(\*"<>ToString[StyleBox[s, FontColor->color, FontWeight->weight, FontVariations->variations], InputForm]<>"\)"
			,
			$UseANSI,
			StringJoin[{colorANSICode[color], weightANSICode[weight], variationsANSICode[variations], s, resetANSICode[]}]
			,
            $UseHTML,
            StringJoin[{
                "<font color=\"" <> colorToHTMLColor[color] <> "\">",
	            With[{
	                b = If[weightHTML[weight] === "",
                        escapeHTML[s],
                        "<b>" <> escapeHTML[s] <> "</b>"
	                ]
	            },    If[variationsHTML[variations] === "",
	                    b,
                        "<u>" <> b <> "</u>"
	                ]
	            ],
                "</font>"
            }],
            $UseHTML5,
            StringJoin[{
                "<span style=\"",
                StringRiffle[
                    Select[
                        {
	                        colorHTML[color],
	                        weightHTML[weight],
	                        variationsHTML[variations]
	                    },
                        # =!= ""&
                    ],
                    ";"
                ],
                "\">",
                escapeHTML[s],
                "</span>"
            }]
            ,
			True,
			s
		]
	] 

escapeHTML[s_] :=
    If[StringContainsQ[s, "<font"],
        s,
	    StringReplace[
	        s,
	        {
	            "\n" -> "<br />",
				"\r" -> "<br />",
				"&" -> "&amp;",
				"<" -> "&lt;",
				">" -> "&gt;",
				"\"" -> "&quot;",
	            "'" -> "&apos;",
	            "\[CapitalOSlash]" -> "&Oslash;",
	            "\[DottedSquare]" -> "&#9633;", 
	            "\[Lambda]" -> "&lambda;",
	            "$" -> "_",
	            "\[Rule]" -> "&rarr;"
	        }
	    ]
    ]


GrayText[args___] :=
	markup[{args}, FontColor->GrayLevel[0.4]]

VeryLightGrayText[args___] :=
	markup[{args}, FontColor->GrayLevel[0.9]]
	
BoldGrayText[args___] :=
	markup[{args}, FontColor->GrayLevel[0.4], FontWeight->Bold]

BoldRedText[args___] :=
	markup[{args}, FontColor->RGBColor[0.66, 0, 0], FontWeight->Bold]

BoldBlackText[args___] :=
	markup[{args}, FontColor->GrayLevel[0.3], FontWeight->Bold]

BoldGreenText[args___] :=
	markup[{args}, FontColor->RGBColor[0.269, 0.538, 0.356], FontWeight->Bold]

InstructionNumberText[args___] :=
	markup[{args}, FontColor->GrayLevel[0.8]]

InstructionNameText[args___] :=
	markup[{args}, FontColor->RGBColor[0.6, 0.4, 0.4], FontWeight->Bold, FontVariations->{"Underline"->True}]

LabelText[args___] :=
	markup[{args}, FontColor->RGBColor[0.9, 0.5, 0.2]]
	
PropertyText[args___] :=
    markup[{args}, FontColor->RGBColor[0.56,0.75,0.86]]

End[]

EndPackage[]

