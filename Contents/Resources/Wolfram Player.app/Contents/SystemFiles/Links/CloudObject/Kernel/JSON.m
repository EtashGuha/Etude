BeginPackage["CloudObject`"]

Begin["`Private`"]

importFromJSON := (
                   Needs["JSONTools`"];
                   importFromJSON = JSONTools`FromJSON
			      )
exportToJSON := (
                 Needs["JSONTools`"];
                 exportToJSON = JSONTools`ToJSON
			    )

End[]

EndPackage[]
