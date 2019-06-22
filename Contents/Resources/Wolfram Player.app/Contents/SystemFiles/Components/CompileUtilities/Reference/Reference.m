
BeginPackage["CompileUtilities`Reference`"]


Reference::usage = ""

CreateReference::usage = "Reference"

ReferenceQ::usage = "Returns True if this is a valid Reference"


Begin["`Private`"] 

Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Reference`ListReference`"]
Needs["CompileUtilities`Reference`AssociationReference`"]


Clear[CreateReference];
CreateReference = Compile`Utilities`Reference`Impl`CreateReference;
Clear[ReferenceQ];
ReferenceQ = Compile`Utilities`Reference`Impl`ReferenceQ;


toBoxes[self_, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"Reference",
		self,
  		BoxForm`GenericIcon[OutputStream],
  		{
  		    BoxForm`SummaryItem[{"id: ", self["refid"]}],
  		    BoxForm`SummaryItem[{"head: ", self["head"]}],
  		    BoxForm`SummaryItem[{"value: ", self["get"]}]
  		},
  		{}, 
  		fmt,
		"Interpretable" -> False
  	]
 
Unprotect[Compile`Utilities`Reference`Impl`GenericReferenceHandler];
Compile`Utilities`Reference`Impl`GenericReferenceHandler[ General, ref_, {"toBoxes", fmt_}] :=
		toBoxes[ ref, fmt];
Compile`Utilities`Reference`Impl`GenericReferenceHandler[ Integer, ref_, {"toBoxes", fmt_}] :=
		toBoxes[ ref, fmt];
Compile`Utilities`Reference`Impl`GenericReferenceHandler[ "Boolean", ref_, {"toBoxes", fmt_}] :=
		toBoxes[ ref, fmt];
		
Compile`Utilities`Reference`Impl`GenericReferenceHandler[ _, ref_, {"toString"}] :=
		ToString[ ref["get"]];
		
Compile`Utilities`Reference`Impl`GenericReferenceHandler[ type_, ref_, args_] :=
		ThrowException[CompilerException[{"Unknown reference access. Type: ", type, " args: ", args}]]
 
Unprotect[ Compile`Utilities`Reference`Impl`ReferenceObject]
Format[ r_Compile`Utilities`Reference`Impl`ReferenceObject, OutputForm] := OutputForm[ r["toString"]]
Format[ r_Compile`Utilities`Reference`Impl`ReferenceObject, InputForm] := OutputForm[ r["toString"]]

Unprotect[ Compile`Utilities`Reference`Reference]
Format[ r_Compile`Utilities`Reference`Reference, OutputForm] := OutputForm[ r["toString"]]
Format[ r_Compile`Utilities`Reference`Reference, InputForm] := OutputForm[ r["toString"]]
	
  	
End[]

EndPackage[]
