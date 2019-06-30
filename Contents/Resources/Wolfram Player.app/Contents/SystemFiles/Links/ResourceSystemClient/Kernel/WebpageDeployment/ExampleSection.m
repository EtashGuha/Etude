
Begin["DeployedResourceShingle`"]
Begin["`Private`"]

DeployedResourceShingle`$webresourcepath=FileNameJoin[{Directory[],"ResourceObjectShingles"}];
DeployedResourceShingle`$webImagePermissions="Private"

exampleSection[rtype_,id_,target_,info_,nbo_NotebookObject]:=exampleSection[rtype,id,target,info,NotebookGet[nbo]]

exampleSection[rtype_,id_,target_,info_,nb_Notebook]:=(
	loadTransmogrify[All];
	shingleexampleSection[rtype,id,target,info,nb]
)

exampleSection[___]:=Missing[]

shingleexampleSection[rtype_,id_,target_,info_,nb_Notebook]:=resourceSystemTransmogrify[rtype,id,prepareExamplesForShingle[rtype,id, target,info,nb]]

resourceSystemTransmogrify[rtype_,id_,nb_]:=(loadTransmogrify[Automatic];
	groupOrphanedInputs[
		ResourceShingleTransmogrify`ResourceShingleTransmogrify[nb,
		transmogrifyRules[rtype],
		ResourceShingleTransmogrify`DefaultParameters ->
			{"FileName" -> resourceshingleimages[rtype,id], "langext" -> ".en"}]
	])




loadTransmogrify[rtype_]:=(loadTransmogrify[rtype]=(Clear["ResourceShingleTransmogrify`*"]; Get[transmogrifyInitFile[rtype]]))
transmogrifyInitFile[rtype_]=FileNameJoin[{$drsDirectory,"Transmogrify","ResourceShingleInit.m"}];

transmogrifyRules[ "Documentation" ] :=
  Module[ { docRules, defaultRules },
      docRules = Import @ transmogrifyRulesFile @ "Documentation";
      defaultRules = Import @ transmogrifyRulesFile @ Automatic;
      mergeXMLTransform[ { defaultRules, docRules }, Last ]
  ];


transmogrifyRules[rtype_]:=Import[transmogrifyRulesFile[rtype]]

transmogrifyRulesFile["Documentation"]:=FileNameJoin[{$drsDirectory,"Transmogrify","documentationPageRules.m"}]

transmogrifyRulesFile[rtype_]:=FileNameJoin[{$drsDirectory,"Transmogrify","TransmogrifyRules.m"}]

$internalWebsiteImages="ResourceObjectShingleImageDump";
resourceshingleimages[_,id_]:=FileNameJoin[{$internalWebsiteImages, StringTake[id,3],id}]


prepareExamplesForShingle[rtype_,id_,target_, info_,nb_]:=
	Replace[replaceExampleNotebookUUIDs[nb, id, target],HoldPattern[ StyleDefinitions -> _] -> (StyleDefinitions -> "Default.nb"), Infinity]

replaceExampleNotebookUUIDs[nb_, id_String, target_String]:=Replace[
	nb, {id->target, ToString[id, InputForm]->ToString[target, InputForm]},Infinity,Heads->True]

replaceExampleNotebookUUIDs[nb_,__]:=nb

groupOrphanedInputs[xml_] := Replace[
  Replace[
   Replace[xml, (eframes :
       XMLElement[
        "div", {"class" -> "example-frame"}, ___]) :> (eframes /.
       "example input" -> "escaped example input"),
    Infinity], (einput :
      XMLElement["table", {"class" -> "example input"}, ___]) :>
    XMLElement["div", {"class" -> "example-frame"}, {einput,

    	XMLElement["table", {"class" -> "example output"}, {
    		XMLElement[ "tr", {}, {XMLElement["td", {"class" -> "in-out"}, {}]}]}]
    	}],
   Infinity],
  "escaped example input" -> "example input", Infinity]




$delayed // ClearAll;
$delayed // Attributes = { HoldAllComplete };
$delayed /: MakeBoxes[ $delayed[ arg_ ], fmt_ ] := MakeBoxes[ arg, fmt ];


mergeEvaluate // ClearAll;

mergeEvaluate[ rules_, f_ ] :=

  Module[ { noDelayed, merged },

      noDelayed =
        Replace[ Normal @ Merge[ rules, $delayed ],
                 {
                     HoldPattern[ lhs_ -> $delayed[ rhs_ ] ] :> lhs -> f @ rhs,
                     HoldPattern[ lhs_ :> $delayed[ { rhs___ } ] ] :>
                       lhs -> With[ { d = Cases[ HoldComplete @ rhs, e_ :> $delayed @ e ] }, f @ d ]
                 },
                 { 1 }
        ];

      merged =
        Replace[ noDelayed,
                 HoldPattern @ Rule[ lhs_, rhs_ /; ! FreeQ[ rhs, $delayed ] ] :>
                   RuleCondition[ (lhs :> rhs) /. $delayed[ arg_ ] :> arg ],
                 { 1 }
        ];

      Association @ merged
  ];



mergeXMLTransform // ClearAll;

mergeXMLTransform[ { }, _ ] := { };

mergeXMLTransform[ { transform_ }, _ ] := transform;

mergeXMLTransform[
    {
        ResourceShingleTransmogrify`Private`XMLTransform[ rules1_List, opts1___ ],
        ResourceShingleTransmogrify`Private`XMLTransform[ rules2_List, opts2___ ],
        rest___
    },
    f_
] :=
  mergeXMLTransform[
      {
          ResourceShingleTransmogrify`Private`XMLTransform[
              Normal @ mergeEvaluate[ { rules1, rules2 }, f ],
              Sequence @@ Normal @ mergeEvaluate[ { opts1, opts2 }, f ]
          ]
      },
      f
  ];



End[]
End[]
