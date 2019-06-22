
BeginPackage["CompileUtilities`Reference`AssociationReference`"]

Begin["`Private`"]


Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Format`"]

	
toBoxes[self_, fmt_] :=
	With[{data = Take[self["get"], Min[Length[self["get"]], 10]]},
		BoxForm`ArrangeSummaryBox[
			"Reference",
			self,
	  		"",
	  		{
	  		    CompileInformationPanel["Association", Normal[data]]
	  		},
	  		{}, 
	  		fmt,
			"Interpretable" -> False
	  	]
	]


Unprotect[Compile`Utilities`Reference`Impl`GenericReferenceHandler];
Compile`Utilities`Reference`Impl`GenericReferenceHandler[ Association, ref_, {"toBoxes", fmt_}] :=
		toBoxes[ ref, fmt];

	
End[]

EndPackage[]
