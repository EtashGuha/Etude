
BeginPackage["CompileUtilities`Reference`ListReference`"]


Begin["`Private`"]


Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Format`"]


toBoxes[self_, fmt_] :=
	Module[{data = Take[self["get"], Min[Length[self["get"]], 10]]},
		data = AssociationThread[ Range[Length[data]]->data];
		BoxForm`ArrangeSummaryBox[
			"Reference",
			self,
	  		"",
	  		{
	  		    CompileInformationPanel["List", Normal[data]]
	  		},
	  		{}, 
	  		fmt,
			"Interpretable" -> False
	  	]
	]


Unprotect[Compile`Utilities`Reference`Impl`GenericReferenceHandler];

Compile`Utilities`Reference`Impl`GenericReferenceHandler[ List, ref_, {"toBoxes", fmt_}] :=
		toBoxes[ ref, fmt];

		
End[]

EndPackage[]
