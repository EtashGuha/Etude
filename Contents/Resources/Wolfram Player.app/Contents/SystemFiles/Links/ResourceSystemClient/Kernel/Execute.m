(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}


BeginPackage["ResourceSystemClient`"]
(* Exported symbols added here with SymbolName::usage *)  
ResourceSystemClient`ResourceInformation

Begin["`Private`"] (* Begin Private Context *) 

ResourceSystemClient`ResourceInformation[args___]:=Catch[resourceInformation[args]]

resourceInformation[resource_ResourceObject,rest___]:=resourceInformation[resourceObjectID[resource],rest]

resourceInformation[id_String,rest___]:=resourceinformation[id,resourceInfo[id],rest]/;MemberQ[$loadedResources, id]

resourceInformation[id_String,rest___]:=With[{loaded=loadResource[id]},
		If[AssociationQ[loaded],
			resourceinformation[id,loaded,rest]
			,
			Throw[$Failed]
		]
	]

resourceInformation[expr_,___]:=(Message[ResourceObject::invro,expr];Throw[$Failed])

resourceinformation[id_String,info_Association,rest___]:=resourceinformation0[getResourceType[info],id,info,rest]

$ReservedProperties={"Properties","DocumentationLink",
	"ExampleNotebook","ExampleNotebookObject",
	"DefinitionNotebook","DefinitionNotebookObject"};

repositoryReservedProperties[ ___ ] := { };
repositoryResourceMetadataLookup[ ___ ] := Missing[ "NotAvailable" ];

resourceinformation0[type_, id_, info_, as_Association, rest___] := With[{prop = as["Property"]},
    loadResourceType @ type;
    If[ MemberQ[ repositoryReservedProperties @ type, prop ]
        ,
        repositoryResourceMetadataLookup[ type, id, info, prop, as, rest ]
        ,
        If[KeyExistsQ[info, prop] || MemberQ[$ReservedProperties, prop],
            resourceMetadataLookup[type, id, info, prop, as, rest]
            ,
            Message[ResourceObject::unkpar, prop];Throw[$Failed]
        ]
    ]
] /; KeyExistsQ[as, "Property"]

resourceMetadataLookup[type_, id_, info_, "Properties",___]:=filterResourceProperties[type,Join[Keys[info],$ReservedProperties, repositoryReservedProperties[type]]]

resourceMetadataLookup[type_, id_, info_, "SourceMetadata",as_,___]:=formatResourceMetadata[Lookup[info,"SourceMetadata"], "SourceMetadata", as["Parameters"]]/;KeyExistsQ[as,"Parameters"]

resourceMetadataLookup[type_, id_, info_, "DocumentationLink",as_,___]:=resourceURL[info]

resourceMetadataLookup[type_, id_, info_, "ExampleNotebook",___]:=getnotebook["Example",id]
resourceMetadataLookup[type_, id_, info_, "DefinitionNotebook",___]:=getnotebook["Definition",id]

resourceMetadataLookup[type_, id_, info_, "ExampleNotebookObject",rest___]:=Lookup[info,"ExampleNotebook"]/;KeyExistsQ[info, "ExampleNotebook"]
resourceMetadataLookup[type_, id_, info_, "DefinitionNotebookObject",rest___]:=Lookup[info,"DefinitionNotebook"]/;KeyExistsQ[info, "DefinitionNotebook"]

resourceMetadataLookup[type_, id_, info_, "ExampleNotebookObject",___]:=Block[{res, nb},
    res=apifun["GetExampleNotebook",{"UUID"->id},
        ResourceObject, resourcerepositoryBase[info]];
    If[KeyExistsQ[res,"ExampleNotebook"],
        setResourceInfo[id, Association["ExampleNotebook"->res["ExampleNotebook"]]];
        res["ExampleNotebook"]
        ,
        Message[ResourceObject::noexamp]
    ]
]

resourceMetadataLookup[type_, id_, info_, "DefinitionNotebookObject",___]:=Block[{res, nb},
    res=apifun["GetDefinitionNotebook",{"UUID"->id}, 
        ResourceObject, resourcerepositoryBase[info]];
    If[KeyExistsQ[res,"DefinitionNotebook"],
        setResourceInfo[id, Association["DefinitionNotebook"->res["DefinitionNotebook"]]];
        res["DefinitionNotebook"]
        ,
        Message[ResourceObject::noexamp]
    ]
]

resourceMetadataLookup[type_, id_, info_, str_,rest___]:=formatResourceMetadata[Lookup[info,str], str, rest]/;KeyExistsQ[info, str]

formatResourceMetadata[elems_,"ContentElements",___]:=DeleteCases[elems,Automatic]

formatResourceMetadata[smd_, "SourceMetadata", {}]:=smd

formatResourceMetadata[smd_, "SourceMetadata", {str_String}]:=Lookup[smd,str]
formatResourceMetadata[smd_, "SourceMetadata", _]:=$Failed

formatResourceMetadata[expr_,___]:=expr

uselessResourceProperties[_]:={"Attributes","MyAccount","DefaultReturnFormat","ContentElementAccessType","DefinitionNotebook","DefinitionNotebookObject"};

filterResourceProperties[rtype_,props_]:=Complement[props,uselessResourceProperties[rtype]]

End[] (* End Private Context *)

EndPackage[]

SetAttributes[{},
   {ReadProtected, Protected}
];