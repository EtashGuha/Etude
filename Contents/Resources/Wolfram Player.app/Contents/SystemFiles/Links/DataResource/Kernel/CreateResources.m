(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["DataResource`"]

Begin["`Private`"] (* Begin Private Context *) 


ResourceSystemClient`Private`repositorystandardizeContentMetadata[$DataResourceType,id_, info_]:=
	standardizeContentMetadataWithElements["DataResource",id, info]

standardizeContentMetadataContentInfo["DataResource",default_, locations_, contentelements_,moreinfo_]:=
	With[{rules={contentElementSize[locations,contentelements, moreinfo,default],
		contentElementAccessType[locations,contentelements, moreinfo,default]}},
		Association[rules]
	]
	
contentElementAccessType[locations_,contentelements_, moreinfo_,default_]:=
	If[default===Automatic,
			"ContentElementAccessType"->"Multipart"
			,
			If[KeyExistsQ[contentelements,default],
				"ContentElementAccessType"->ToString[Head[contentelements[default]]]
				,
				If[KeyExistsQ[moreinfo,default],
					"ContentElementAccessType"->ToString[Head[moreinfo[default]]],
					"ContentElementAccessType"->Missing["NotAvailable"]
				]
			]
		]


ResourceSystemClient`Private`repositorysaveresourceobject[$DataResourceType,info_]:=saveresourceobjectwithelements[info]


End[] (* End Private Context *)

EndPackage[]



SetAttributes[{},
   {ReadProtected, Protected}
];