

BeginPackage["CompileAST`Export`Format`Information`WhiteSpace`"]


Begin["`Private`"] 

Needs["CompileAST`Export`Format`Information`Character`"]


whiteSpaceQ[string_String] :=
   SameQ[
      DeleteCases[
         Characters[string],
         Alternatives[
            "\t",
            "\n",
            " ",
            "\[InvisibleSpace]",
            "\[VeryThinSpace]",
            "\[ThinSpace]",
            "\[MediumSpace]",
            "\[ThickSpace]",
            "\[NegativeVeryThinSpace]",
            "\[NegativeThinSpace]",
            "\[IndentingNewLine]",
            "\[NegativeMediumSpace]",
            "\[NegativeThickSpace]",
            "\r",
            "\[NoBreak]",
            "\[NonBreakingSpace]",
            "\[Continuation]",
            "\[SpaceIndicator]",
            "\[RoundSpaceIndicator]",
            "\[AlignmentMarker]",
            "",
            "\[LineSeparator]",
            "\[ParagraphSeparator]"
         ]
      ],
      {}
   ];
whiteSpaceQ[other___] := False;

SyntaxCharacterInformation[tok_, "WhiteSpaceQ"] :=
	TrueQ[With[{info = SyntaxCharacterInformation[tok]},
		If[FailureQ[info],
			False,
			whiteSpaceQ[info["Character"]]
		]
	]]

End[]
EndPackage[]
