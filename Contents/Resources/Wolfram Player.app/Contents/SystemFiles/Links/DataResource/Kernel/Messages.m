(* Wolfram Language Package *)

BeginPackage["DataResource`"]
(* Exported symbols added here with SymbolName::usage *)  

Begin["`Private`"] (* Begin Private Context *) 

ResourceData::norsys="The ResourceSystemClient paclet could not be found. `1` will not behave correctly."
ResourceData::nopacl="The `2` paclet could not be found. `1` will not behave correctly."
ResourceData::cloudc="You must connect to the Wolfram cloud to access the resource data."

ResourceObject::depcf="The content of element `1` could not be copied to the cloud."
ResourceObject::nbood="The resource content could not be collected from the notebook, "<>
	"possibly due to the original Wolfram Language kernel being terminated. Please try again with a new creation notebook."

ResourceData::rtype="The ResourceType of the specific resource is not supported by ResourceData."

ResourceObject::craddex="Please create the resource object using the \"CREATE\" button before adding examples"

ResourceObject::contfmt="The content of the data resource should be a valid Association."

End[] (* End Private Context *)

EndPackage[]