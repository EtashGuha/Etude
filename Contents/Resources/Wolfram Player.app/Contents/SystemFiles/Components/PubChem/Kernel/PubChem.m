Begin["PubChemAPI`"]

ServiceExecute::niden = "Only one identifier is allowed";

Begin["`Private`"]

(******************************* PubChem *************************************)

(* Authentication information *)

pubchemdata[]:={
        "ServiceName"         -> "PubChem", 
        "URLFetchFun"        :> (With[{params=Lookup[{##2},"Parameters",{}]},
        							(URLFetch[#1,"ContentData",Sequence@@FilterRules[{##2},Except["Parameters"|"Headers"]],
        							"Parameters" -> params,
        							"Headers" -> {}])
        						]&),
         "ClientInfo"		   -> {},						                                               
         "Gets"                -> {"CompoundDescription","CompoundSynonyms","CompoundSID","CompoundAID","CompoundCID","CompoundCrossReferences","CompoundProperties","CompoundImage","CompoundFullRecords","CompoundAssaySummary",
         							"SubstanceSynonyms","SubstanceSID","SubstanceAID","SubstanceCID","SubstanceImage","SubstanceFullRecords","SubstanceAssaySummary","SubstanceCrossReferences","CompoundSDF","SubstanceSDF"},
         "Posts"               -> {},
         "RawGets"             -> {"RawCompounds", "RawSubstances","RawCompoundImage","RawSubstanceImage","RawCompoundSDF","RawSubstanceSDF"},
         "RawPosts"            -> {"RawCompoundsPost","RawCompoundImagePost","RawCompoundSDFPost"},
         "Information"         -> "Wolfram Language connection to Pubchem API"
}

(*Raw*)

pubchemimport[rawdata_]:=FromCharacterCode[rawdata]

pubchemdata["RawCompounds"] := {
        "URL"				-> (ToString@StringForm["https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/`1`/`2`/`3`/`4`/", #1,StringReplace[#2," ":>"%20"],#3,#4]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"namespace","identifiers","operation","format"},
        "Parameters"		-> Join[searchparams,{"record_type","cids_type","sids_type","aids_type"}],
        "RequiredParameters"-> {"namespace","identifiers","operation","format"},
        "ResultsFunction"	-> pubchemimport
        }

pubchemdata["RawCompoundsPost"] := {
        "URL"				-> (ToString@StringForm["https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/`1`/`2`/`3`/", #1,#2,#3]&),
        "HTTPSMethod"		-> "POST",
        "BodyData"			-> {"ParameterlessBodyData"->"identifiers"},
        "PathParameters"	-> {"namespace","operation","format"},
        "Parameters"		-> {},
        "RequiredParameters"-> {"identifiers","namespace","operation","format"},
        "ResultsFunction"	-> pubchemimport
        }

pubchemdata["RawSubstances"] := {
        "URL"				-> (ToString@StringForm["https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/`1`/`2`/`3`/`4`/", #1,StringReplace[#2," ":>"%20"],#3,#4]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"namespace","identifiers","operation","format"},
        "Parameters"		-> {"cids_type","sids_type","aids_type"},
        "RequiredParameters"-> {"namespace","identifiers","operation","format"},
        "ResultsFunction"	-> pubchemimport
        }

pubchemdata["RawCompoundImage"] := {
        "URL"				-> (ToString@StringForm["https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/`1`/`2`/PNG/", #1,StringReplace[#2," ":>"%20"]]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"namespace","identifiers"},
        "Parameters"		-> {"record_type","image_size"},
        "RequiredParameters"-> {"namespace","identifiers"},
        "ResultsFunction"	-> pubchemimport
        }

pubchemdata["RawCompoundImagePost"] := {
        "URL"				-> (ToString@StringForm["https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/`1`/PNG/", #1]&),
        "HTTPSMethod"		-> "POST",
        "BodyData"			-> {"ParameterlessBodyData"->"identifiers"},
        "PathParameters"	-> {"namespace"},
        "Parameters"		-> {"record_type","image_size"},
        "RequiredParameters"-> {"identifiers","namespace"},
        "ResultsFunction"	-> pubchemimport
        }

pubchemdata["RawSubstanceImage"] := {
        "URL"				-> (ToString@StringForm["https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/`1`/`2`/PNG/", #1,StringReplace[#2," ":>"%20"]]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"namespace","identifiers"},
        "Parameters"		-> {"record_type","image_size"},
        "RequiredParameters"-> {"namespace","identifiers"},
        "ResultsFunction"	-> pubchemimport
        }
        
pubchemdata["RawCompoundSDF"] := {
        "URL"				-> (ToString@StringForm["https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/`1`/`2`/SDF/", #1,StringReplace[#2," ":>"%20"]]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"namespace","identifiers"},
        "Parameters"		-> {"record_type"},
        "RequiredParameters"-> {"namespace","identifiers"},
        "ResultsFunction"	-> pubchemimport
        }
        
pubchemdata["RawCompoundSDFPost"] := {
        "URL"				-> (ToString@StringForm["https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/`1`/SDF/", #1]&),
        "HTTPSMethod"		-> "POST",
        "BodyData"			-> {"ParameterlessBodyData"->"identifiers"},
        "PathParameters"	-> {"namespace"},
        "Parameters"		-> {"record_type"},
        "RequiredParameters"-> {"identifiers","namespace"},
        "ResultsFunction"	-> pubchemimport
        }    
        
pubchemdata["RawSubstanceSDF"] := {
        "URL"				-> (ToString@StringForm["https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/`1`/`2`/SDF/", #1,StringReplace[#2," ":>"%20"]]&),
        "HTTPSMethod"		-> "GET",
        "PathParameters"	-> {"namespace","identifiers"},
        "Parameters"		-> {"record_type"},
        "RequiredParameters"-> {"namespace","identifiers"},
        "ResultsFunction"	-> pubchemimport
        }
(*Cooked*)

allxrefs={"RegistryID", "RN", "PubMedID", "MMDBID", "DBURL", "SBURL","ProteinGI", "NucleotideGI", "TaxonomyID", "MIMID", "GeneID","ProbeID", "PatentID", "SourceName", "SourceCategory"}

allprops={"MolecularFormula", "MolecularWeight", "CanonicalSMILES", "IsomericSMILES", "InChI", "InChIKey", "IUPACName", "XLogP", "ExactMass", "MonoisotopicMass", "TPSA",
	"Complexity", "Charge", "HBondDonorCount", "HBondAcceptorCount", "RotatableBondCount", "HeavyAtomCount", "IsotopeAtomCount", "AtomStereoCount", "DefinedAtomStereoCount",
	"UndefinedAtomStereoCount", "BondStereoCount", "DefinedBondStereoCount", "UndefinedBondStereoCount", "CovalentUnitCount", "Volume3D", "XStericQuadrupole3D", "YStericQuadrupole3D",
	"ZStericQuadrupole3D", "FeatureCount3D", "FeatureAcceptorCount3D", "FeatureDonorCount3D", "FeatureAnionCount3D", "FeatureCationCount3D", "FeatureRingCount3D", "FeatureHydrophobeCount3D",
	"ConformerModelRMSD3D", "EffectiveRotorCount3D", "ConformerCount3D", "Fingerprint2D"}

searchparams={(*formula*)"AllowOtherElements",(*super and substrutructure*)"MatchIsotopes", "MatchCharges", "MatchTautomers", "RingsNotEmbedded", "SingleDoubleBondsMatch", "ChainsMatchRings", "StripHydrogen", "Stereo",(*similarity (minimum Tanimoto score)*)"Threshold"}

cidtype={"Original" -> "original", "Parent" -> "parent", "Component" -> "component", "Similar2D" -> "similar_2d", "Similar3D" -> "similar_3d", "SameStereo" -> "same_stereo", "SameIsotopes" -> "same_isotopes", 
 "SameConnectivity" -> "same_connectivity", "SameTautomer" -> "same_tautomer", "SameParent" -> "same_parent", "SameParentStereo" -> "same_parent_stereo", "SameParentIsotopes" -> "same_parent_isotopes", 
 "SameParentConnectivity" -> "same_parent_connectivity", "SameParentTautomer" -> "same_parent_tautomer"}
sidtype={"Original" -> "original", "SameExact" -> "same_exact", "SameStereo" -> "same_stereo", "SameIsotopes" -> "same_isotopes", "SameConnectivity" -> "same_connectivity", "SameTautomer" -> "same_tautomer",
 "SameParent" -> "same_parent", "SameParentStereo" -> "same_parent_stereo", "SameParentIsotopes" -> "same_parent_isotopes", "SameParentConnectivity" -> "same_parent_connectivity", "SameParentTautomer" -> "same_parent_tautomer"}

camelCase[text_] := Module[{split, partial}, (
    split = StringSplit[text, {" ","_","-"}];
    partial = Prepend[Rest[Characters[#]], ToUpperCase[Characters[#][[1]]]] & /@ split;
    StringJoin[partial]
    )]

pubchemcookeddata[req:"CompoundDescription"|"CompoundSynonyms"|"CompoundSID"|"CompoundAID"|"CompoundCID", id_,args_]:=Block[{rawdata,newparams,input,compound,namespace,identifiers,operation,params={},invalidParameters,withCamelTitles,search,interpret},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[Join[{"CompoundID","Name","InChI","InChIKey","SMILES","Formula",Method,"AIDType","SIDType","CIDType","InterpretEntities"},searchparams],#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"PubChem"]&/@invalidParameters;
		Throw[$Failed]
	)];
	input = Select[Keys[newparams],MemberQ[{"CompoundID","Name","InChI","InChIKey","SMILES","Formula"},#]&]; 
	If[Length[input]>1,
	(
		Message[ServiceExecute::niden];
		Throw[$Failed]
	)];
	
	Switch[input,
	{"CompoundID"},
	(
		newparams=newparams/.{Rule["CompoundID",a_]:>Rule["Compound",{"CompoundID",a}]} (*FALTAAA*)
	),{"Name"},
	(
		newparams=newparams/.{Rule["Name",a_]:>Rule["Compound",{"Name",a}]}
	),{"InChI"},
	(
		newparams=newparams/.{Rule["InChI",a_]:>Rule["Compound",{"InChI",a}]}
	),{"InChIKey"},
	(
		newparams=newparams/.{Rule["InChIKey",a_]:>Rule["Compound",{"InChIKey",a}]}
	),{"SMILES"},
	(
		newparams=newparams/.{Rule["SMILES",a_]:>Rule["Compound",{"SMILES",a}]}
	),{"Formula"},
	(
		newparams=newparams/.{Rule["Formula",a_]:>Rule["Compound",a]}
	),{},
	(
		Message[ServiceExecute::nparam,"CompoundID, Name, InChI, InChIKey or SMILES","PubChem"];
		Throw[$Failed]
	)];
	
	
	If[KeyExistsQ[newparams,Method],
	(
		If[!StringMatchQ[Method/.newparams, "Similarity2DSearch"|"Similarity3DSearch"|"SubstructureSearch"|"SuperstructureSearch"|"FormulaSearch"],
		(	
			Message[ServiceExecute::nval,"Method","PubChem"];
			Throw[$Failed]
		)];
		search=(Method/.newparams)/.{"Similarity2DSearch"->"fastsimilarity_2d","Similarity3DSearch"->"fastsimilarity_3d","SubstructureSearch"->"fastsubstructure","SuperstructureSearch"->"fastsuperstructure","FormulaSearch"->"fastformula"};
		(*search parameters*)
		If[KeyExistsQ[newparams,#],
		(
			If[!MemberQ[{True,False},#/.newparams],
			(	
				Message[ServiceExecute::nval,#,"PubChem"];
				Throw[$Failed]
			)];
			params=Append[params,Rule[#,(#/.newparams)/.{True->"true",False->"false"}]]
		)]&/@searchparams[[1;;8]];
		If[KeyExistsQ[newparams,"Stereo"],
		(
			If[!StringMatchQ["Stereo"/.newparams, "ignore"|"exact"|"relative"|"nonconflicting",IgnoreCase->True],
			(	
				Message[ServiceExecute::nval,"Stereo","PubChem"];
				Throw[$Failed]
			)];
			params=Append[params,Rule["Stereo",("Stereo"/.newparams)]]
		)];
		If[KeyExistsQ[newparams,"Threshold"],
		(
			If[!IntegerQ["Threshold"/.newparams],
			(	
				Message[ServiceExecute::nval,"Threshold","PubChem"];
				Throw[$Failed]
			)];
			params=Append[params,Rule["Threshold",ToString[("Threshold"/.newparams)]]]
		)];
		
		If[MatchQ[search,"fastformula"],
		(
			If[KeyExistsQ[newparams,"Compound"],
			(
				If[!StringQ["Compound"/.newparams],
				(	
					Message[ServiceExecute::nval,input[[1]],"PubChem"];
					Throw[$Failed]
				)];
				namespace=search;
				identifiers="Compound"/.newparams;
			),
			(
				Message[ServiceExecute::nparam,input[[1]],"PubChem"];
				Throw[$Failed]
			)];
		),
		(
			If[KeyExistsQ[newparams,"Compound"],
			(
				compound="Compound"/.newparams;
				If[!MatchQ[compound,List[_String,_String|_Integer]],
				(	
					Message[ServiceExecute::nval,input[[1]],"PubChem"];
					Throw[$Failed]
				)];
				
				If[!StringMatchQ[compound[[1]], "CompoundID"|"SMILES"|"InChI"],
				(	
					Message[ServiceExecute::nval,input[[1]],"PubChem"];
					Throw[$Failed]
				)];
				namespace=StringJoin[search,"/",(compound[[1]]/.{"CompoundID"->"cid","SMILES"->"smiles","InChI"->"inchi"})];
				identifiers=ToString[compound[[2]]];
			),
			(
				Message[ServiceExecute::nparam,input[[1]],"PubChem"];
				Throw[$Failed]
			)];
		)]
	),
	(
		If[KeyExistsQ[newparams,"Compound"],
		(
			compound="Compound"/.newparams;
			If[!MatchQ[compound,List[_String,_String|_Integer|_Entity|List[__Entity]|List[__String]|List[__Integer]]],
			(	
				Message[ServiceExecute::nval,input[[1]],"PubChem"];
				Throw[$Failed]
			)];
			If[!((StringMatchQ[compound[[1]], "CompoundID"|"SMILES"|"InChIKey"|"Name"|"InChI"]&&MatchQ[compound,List[_String,_String|_Integer|List[__String]|List[__Integer]]])||(MatchQ[compound,List["Name",Entity["Chemical",_]]])||(MatchQ[compound,List["Name",List[Entity["Chemical", _] ..]]])),
			(	
				Message[ServiceExecute::nval,input[[1]],"PubChem"];
				Throw[$Failed]
			)];
			compound={compound[[1]]/.{"CompoundID"->"cid","SMILES"->"smiles","Name"->"name","InChI"->"inchi","InChIKey"->"inchikey"},compound[[2]]};
			If[MatchQ[compound,List["name",Entity["Chemical",_]]],
			(	
				namespace=compound[[1]];
				identifiers=ToString[compound[[2]]["Name"]]
			)];
			If[MatchQ[compound,List["name",List[Entity["Chemical",_]..]]],
			(	
				namespace=compound[[1]];
				identifiers=ToString[#["Name"]]&/@compound[[2]]
			)];
			If[MatchQ[compound,List["cid"|"smiles"|"inchikey"|"name"|"inchi",_String|_Integer]],
			(	
				namespace=compound[[1]];
				identifiers=ToString[compound[[2]]]
			)];
			If[MatchQ[compound,List["cid"|"smiles"|"inchikey",List[__String]|List[__Integer]]],
			(	
				namespace=compound[[1]];
				identifiers=StringJoin[Riffle[ToString/@compound[[2]],","]]
			)];
			If[MatchQ[compound,List["name"|"inchi",List[__String]|List[__Integer]]],
			(	
				namespace=compound[[1]];
				identifiers=ToString/@compound[[2]]
			)];
		),
		(
			Message[ServiceExecute::nparam,input[[1]],"PubChem"];
			Throw[$Failed]
		)];
	)];
	If[KeyExistsQ[newparams,"InterpretEntities"],
	(
		If[!MemberQ[{True,False},"InterpretEntities"/.newparams],
		(	
			Message[ServiceExecute::nval,"InterpretEntities","PubChem"];
			Throw[$Failed]
		)];
		interpret="InterpretEntities"/.newparams
	),
	(
		interpret=False
	)];
	Switch[req,
		"CompoundDescription", operation="description",
		"CompoundSynonyms",operation="synonyms",
		"CompoundSID",
			operation="sids";
			If[KeyExistsQ[newparams,"SIDType"],
			(
				If[!StringMatchQ[ToString["SIDType"/.newparams], "All"|"Standardized"|"Component"],
				(	
					Message[ServiceExecute::nval,"SIDType","PubChem"];
					Throw[$Failed]
				)];
				params=Append[params,Rule["sids_type",ToString["SIDType"/.newparams]/.{"All"->"all","Standardized"->"standardized","Component"->"component"}]]
			)];,
		"CompoundAID",
			operation="aids";
			If[KeyExistsQ[newparams,"AIDType"],
			(
				If[!StringMatchQ[ToString["AIDType"/.newparams], "All"|"Active"|"Inactive"],
				(	
					Message[ServiceExecute::nval,"AIDType","PubChem"];
					Throw[$Failed]
				)];
				params=Append[params,Rule["aids_type",ToString["AIDType"/.newparams]/.{"All"->"all","Active"->"active","Inactive"->"inactive"}]]
			)];,
		"CompoundCID",
			operation="cids";
			If[KeyExistsQ[newparams,"CIDType"],
			(
				If[!StringMatchQ[ToString["CIDType"/.newparams], (Alternatives @@ cidtype[[All, 1]])],
				(	
					Message[ServiceExecute::nval,"CIDType","PubChem"];
					Throw[$Failed]
				)];
				params=Append[params,Rule["cids_type",ToString["CIDType"/.newparams]/.cidtype]]
			)];
			
	];
	Switch[namespace,
	"fastsimilarity_2d/inchi"|"fastsimilarity_3d/inchi"|"fastsubstructure/inchi"|"fastsuperstructure/inchi",
	(
		rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompoundsPost",{"namespace" -> namespace, "identifiers" -> StringJoin["inchi=", identifiers, Sequence @@ (params /. {Rule[a_, b_] :> StringJoin["&", a, "=", b]})], "operation" -> operation, "format" -> "JSON"}],"RawJSON"]]
	),"inchi",
	(
		If[MatchQ[compound,List["inchi",List[__]]],
		(	
			rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompoundsPost",{"namespace" -> namespace, "identifiers" -> StringJoin["inchi=", #, Sequence @@ (params /. {Rule[a_, b_] :> StringJoin["&", a, "=", b]})], "operation" -> operation, "format" -> "JSON"}],"RawJSON"]]& /@ identifiers
		),
		(
			rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompoundsPost",{"namespace" -> namespace, "identifiers" -> StringJoin["inchi=", identifiers, Sequence @@ (params /. {Rule[a_, b_] :> StringJoin["&", a, "=", b]})], "operation" -> operation, "format" -> "JSON"}],"RawJSON"]]
		)];
	),"name",
	(
		If[MatchQ[compound,List["name",List[__]]],
		(	
			rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompounds",Join[{"namespace" -> namespace, "identifiers" -> #, "operation" -> operation, "format" -> "JSON"},params]],"RawJSON"]]&/@identifiers
		),
		(
			rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompounds",Join[{"namespace" -> namespace, "identifiers" -> identifiers, "operation" -> operation, "format" -> "JSON"},params]],"RawJSON"]]
		)];
	),_,
	(
		rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompounds",Join[{"namespace" -> namespace, "identifiers" -> identifiers, "operation" -> operation, "format" -> "JSON"},params]],"RawJSON"]]
	)];	
	If[MatchQ[compound,List["name"|"inchi",List[__]]],
	(
		If[MatchQ[#,$Failed],
   		(
    	  	Message[ServiceExecute::serrormsg,""];
    	   	Throw[$Failed]
 		)]&/@rawdata;
 		If[KeyExistsQ[#,"Fault"],
   		(
    	  	Message[ServiceExecute::serrormsg,"Message" /. ("Fault" /. #)];
    	   	Throw[$Failed]
 		)]&/@rawdata;
 	),
 	(
		If[MatchQ[rawdata,$Failed],
   		(
    	  	Message[ServiceExecute::serrormsg,""];
    	   	Throw[$Failed]
 		)];
 		If[KeyExistsQ[rawdata,"Fault"],
   		(
    	  	Message[ServiceExecute::serrormsg,"Message" /. ("Fault" /. rawdata)];
    	   	Throw[$Failed]
 		)];
 	)];
 	rawdata = FixedPoint[Normal, rawdata];
 	Switch[req,
		"CompoundDescription"|"CompoundSynonyms"|"CompoundSID"|"CompoundAID",
			rawdata = "Information" /. ("InformationList" /. rawdata);
			withCamelTitles=rawdata/.{"SID"->"SubstanceID","CID"->"CompoundID","AID"->"AssayID"};
			If[interpret,withCamelTitles=withCamelTitles/.{Rule["Title",a_String]:>Rule["Title",If[MatchQ[b=Interpreter["Chemical"][a],Failure[__]],a,b]]}];
			Dataset[Association @@@ withCamelTitles],
		"CompoundCID",
			rawdata = "IdentifierList" /. rawdata;
			withCamelTitles=rawdata/.{"SID"->"SubstanceID","CID"->"CompoundID","AID"->"AssayID"};
			If[MatchQ[compound,List["name"|"inchi",List[__]]],
			(
				Dataset[Association /@ withCamelTitles]
 			),
			(
				Dataset[Association @@ withCamelTitles]
			)]
	]
]

pubchemcookeddata["CompoundCrossReferences", id_,args_]:=Block[{rawdata,newparams,input,compound,namespace,identifiers,xref,xref2,params={},invalidParameters,withCamelTitles,search},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[Join[{"CompoundID","Name","InChI","InChIKey","SMILES","Formula","CrossReference",Method},searchparams],#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"PubChem"]&/@invalidParameters;
		Throw[$Failed]
	)];
	input = Select[Keys[newparams],MemberQ[{"CompoundID","Name","InChI","InChIKey","SMILES","Formula"},#]&]; 
	If[Length[input]>1,
	(
		Message[ServiceExecute::niden];
		Throw[$Failed]
	)];
	
	Switch[input,
	{"CompoundID"},
	(
		newparams=newparams/.{Rule["CompoundID",a_]:>Rule["Compound",{"CompoundID",a}]} (*FALTAAA*)
	),{"Name"},
	(
		newparams=newparams/.{Rule["Name",a_]:>Rule["Compound",{"Name",a}]}
	),{"InChI"},
	(
		newparams=newparams/.{Rule["InChI",a_]:>Rule["Compound",{"InChI",a}]}
	),{"InChIKey"},
	(
		newparams=newparams/.{Rule["InChIKey",a_]:>Rule["Compound",{"InChIKey",a}]}
	),{"SMILES"},
	(
		newparams=newparams/.{Rule["SMILES",a_]:>Rule["Compound",{"SMILES",a}]}
	),{"Formula"},
	(
		newparams=newparams/.{Rule["Formula",a_]:>Rule["Compound",a]}
	),{},
	(
		Message[ServiceExecute::nparam,"CompoundID, Name, InChI, InChIKey or SMILES","PubChem"];
		Throw[$Failed]
	)];
	
	If[KeyExistsQ[newparams,Method],
	(
		If[!StringMatchQ[Method/.newparams, "Similarity2DSearch"|"Similarity3DSearch"|"SubstructureSearch"|"SuperstructureSearch"|"FormulaSearch"],
		(	
			Message[ServiceExecute::nval,"Method","PubChem"];
			Throw[$Failed]
		)];
		search=(Method/.newparams)/.{"Similarity2DSearch"->"fastsimilarity_2d","Similarity3DSearch"->"fastsimilarity_3d","SubstructureSearch"->"fastsubstructure","SuperstructureSearch"->"fastsuperstructure","FormulaSearch"->"fastformula"};
		(*search parameters*)
		If[KeyExistsQ[newparams,#],
		(
			If[!MemberQ[{True,False},#/.newparams],
			(	
				Message[ServiceExecute::nval,#,"PubChem"];
				Throw[$Failed]
			)];
			params=Append[params,Rule[#,(#/.newparams)/.{True->"true",False->"false"}]]
		)]&/@searchparams[[1;;8]];
		If[KeyExistsQ[newparams,"Stereo"],
		(
			If[!StringMatchQ["Stereo"/.newparams, "ignore"|"exact"|"relative"|"nonconflicting",IgnoreCase->True],
			(	
				Message[ServiceExecute::nval,"Stereo","PubChem"];
				Throw[$Failed]
			)];
			params=Append[params,Rule["Stereo",("Stereo"/.newparams)]]
		)];
		If[KeyExistsQ[newparams,"Threshold"],
		(
			If[!IntegerQ["Threshold"/.newparams],
			(	
				Message[ServiceExecute::nval,"Threshold","PubChem"];
				Throw[$Failed]
			)];
			params=Append[params,Rule["Threshold",ToString[("Threshold"/.newparams)]]]
		)];
		
		If[MatchQ[search,"fastformula"],
		(
			If[KeyExistsQ[newparams,"Compound"],
			(
				If[!StringQ["Compound"/.newparams],
				(	
					Message[ServiceExecute::nval,input[[1]],"PubChem"];
					Throw[$Failed]
				)];
				namespace=search;
				identifiers="Compound"/.newparams;
			),
			(
				Message[ServiceExecute::nparam,input[[1]],"PubChem"];
				Throw[$Failed]
			)];
		),
		(
			If[KeyExistsQ[newparams,"Compound"],
			(
				compound="Compound"/.newparams;
				If[!MatchQ[compound,List[_String,_String|_Integer]],
				(	
					Message[ServiceExecute::nval,input[[1]],"PubChem"];
					Throw[$Failed]
				)];
				
				If[!StringMatchQ[compound[[1]], "CompoundID"|"SMILES"|"InChI"],
				(	
					Message[ServiceExecute::nval,input[[1]],"PubChem"];
					Throw[$Failed]
				)];
				namespace=StringJoin[search,"/",(compound[[1]]/.{"CompoundID"->"cid","SMILES"->"smiles","InChI"->"inchi"})];
				identifiers=ToString[compound[[2]]];
			),
			(
				Message[ServiceExecute::nparam,input[[1]],"PubChem"];
				Throw[$Failed]
			)];
		)]
	),
	(
		If[KeyExistsQ[newparams,"Compound"],
		(
			compound="Compound"/.newparams;
			If[!MatchQ[compound,List[_String,_String|_Integer|_Entity|List[__Entity]|List[__String]|List[__Integer]]],
			(	
				Message[ServiceExecute::nval,input[[1]],"PubChem"];
				Throw[$Failed]
			)];
			If[!((StringMatchQ[compound[[1]], "CompoundID"|"SMILES"|"InChIKey"|"Name"|"InChI"]&&MatchQ[compound,List[_String,_String|_Integer|List[__String]|List[__Integer]]])||(MatchQ[compound,List["Name",Entity["Chemical",_]]])||(MatchQ[compound,List["Name",List[Entity["Chemical", _] ..]]])),
			(	
				Message[ServiceExecute::nval,input[[1]],"PubChem"];
				Throw[$Failed]
			)];
			compound={compound[[1]]/.{"CompoundID"->"cid","SMILES"->"smiles","Name"->"name","InChI"->"inchi","InChIKey"->"inchikey"},compound[[2]]};
			If[MatchQ[compound,List["name",Entity["Chemical",_]]],
			(	
				namespace=compound[[1]];
				identifiers=ToString[compound[[2]]["Name"]]
			)];
			If[MatchQ[compound,List["name",List[Entity["Chemical",_]..]]],
			(	
				namespace=compound[[1]];
				identifiers=ToString[#["Name"]]&/@compound[[2]]
			)];
			If[MatchQ[compound,List["cid"|"smiles"|"inchikey"|"name"|"inchi",_String|_Integer]],
			(	
				namespace=compound[[1]];
				identifiers=ToString[compound[[2]]]
			)];
			If[MatchQ[compound,List["cid"|"smiles"|"inchikey",List[__String]|List[__Integer]]],
			(	
				namespace=compound[[1]];
				identifiers=StringJoin[Riffle[ToString/@compound[[2]],","]]
			)];
			If[MatchQ[compound,List["name"|"inchi",List[__String]|List[__Integer]]],
			(	
				namespace=compound[[1]];
				identifiers=ToString/@compound[[2]]
			)];
		),
		(
			Message[ServiceExecute::nparam,input[[1]],"PubChem"];
			Throw[$Failed]
		)];
	)];
	If[KeyExistsQ[newparams,"CrossReference"],
	(
		If[!(StringQ["CrossReference"/.newparams]||MatchQ["CrossReference"/.newparams,List[__String] | All]),
		(	
			Message[ServiceExecute::nval,"CrossReference","PubChem"];
			Throw[$Failed]
		)];
		If[!MatchQ["CrossReference"/.newparams,All],
		(
			xref=Flatten[{"CrossReference"/.newparams}];
			If[!(And@@StringMatchQ[xref, Alternatives@@allxrefs]),
			(	
				Message[ServiceExecute::nval,"CrossReference","PubChem"];
				Throw[$Failed]
			)];		
			xref2=StringJoin[Prepend[Riffle[xref,","],"xrefs/"]]
		),
		(
			xref=allxrefs;
			xref2=StringJoin[Prepend[Riffle[xref,","],"xrefs/"]]
		)]
	),
	(
		xref=allxrefs;
		xref2=StringJoin[Prepend[Riffle[xref,","],"xrefs/"]]
	)];
	
	Switch[namespace,
	"fastsimilarity_2d/inchi"|"fastsimilarity_3d/inchi"|"fastsubstructure/inchi"|"fastsuperstructure/inchi",
	(
		rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompoundsPost",{"namespace" -> namespace, "identifiers" -> StringJoin["inchi=", identifiers, Sequence @@ (params /. {Rule[a_, b_] :> StringJoin["&", a, "=", b]})], "operation" -> xref2, "format" -> "JSON"}],"RawJSON"]]
	),"inchi",
	(
		If[MatchQ[compound,List["inchi",List[__]]],
		(	
			rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompoundsPost",{"namespace" -> namespace, "identifiers" -> StringJoin["inchi=", #, Sequence @@ (params /. {Rule[a_, b_] :> StringJoin["&", a, "=", b]})], "operation" -> xref2, "format" -> "JSON"}],"RawJSON"]]& /@ identifiers
		),
		(
			rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompoundsPost",{"namespace" -> namespace, "identifiers" -> StringJoin["inchi=", identifiers, Sequence @@ (params /. {Rule[a_, b_] :> StringJoin["&", a, "=", b]})], "operation" -> xref2, "format" -> "JSON"}],"RawJSON"]]
		)];
	),"name",
	(
		If[MatchQ[compound,List["name",List[__]]],
		(	
			rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompounds",Join[{"namespace" -> namespace, "identifiers" -> #, "operation" -> xref2, "format" -> "JSON"},params]],"RawJSON"]]&/@identifiers
		),
		(
			rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompounds",Join[{"namespace" -> namespace, "identifiers" -> identifiers, "operation" -> xref2, "format" -> "JSON"},params]],"RawJSON"]]
		)];
	),_,
	(
		rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompounds",Join[{"namespace" -> namespace, "identifiers" -> identifiers, "operation" -> xref2, "format" -> "JSON"},params]],"RawJSON"]]
	)];	
	If[MatchQ[compound,List["name"|"inchi",List[__]]],
	(
		If[MatchQ[#,$Failed],
   		(
    	  	Message[ServiceExecute::serrormsg,""];
    	   	Throw[$Failed]
 		)]&/@rawdata;
 		If[KeyExistsQ[#,"Fault"],
   		(
    	  	Message[ServiceExecute::serrormsg,"Message" /. ("Fault" /. #)];
    	   	Throw[$Failed]
 		)]&/@rawdata;
 	),
 	(
		If[MatchQ[rawdata,$Failed],
   		(
    	  	Message[ServiceExecute::serrormsg,""];
    	   	Throw[$Failed]
 		)];
 		If[KeyExistsQ[rawdata,"Fault"],
   		(
    	  	Message[ServiceExecute::serrormsg,"Message" /. ("Fault" /. rawdata)];
    	   	Throw[$Failed]
 		)];
 	)];
	rawdata = FixedPoint[Normal, rawdata];
 	rawdata = "Information" /. ("InformationList" /. rawdata);
	withCamelTitles=Association @@@(rawdata/.{"CID"->"CompoundID"});
	withCamelTitles=Join[#, Association[Replace[DeleteCases[Sort[Append[xref, "CompoundID"]], Alternatives @@ Sort[Keys[#]]], {a_ :> Rule[a, Missing["NotAvailable"]]}, 1]]] & /@ withCamelTitles;
	Dataset[withCamelTitles][All,Flatten[{"CompoundID",xref}]]
]

pubchemcookeddata["CompoundProperties", id_,args_]:=Block[{rawdata,newparams,input,compound,namespace,identifiers,prop,prop2,params={},invalidParameters,withCamelTitles,search},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[Join[{"CompoundID","Name","InChI","InChIKey","SMILES","Formula","Property",Method},searchparams],#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"PubChem"]&/@invalidParameters;
		Throw[$Failed]
	)];
	
	input = Select[Keys[newparams],MemberQ[{"CompoundID","Name","InChI","InChIKey","SMILES","Formula"},#]&]; 
	If[Length[input]>1,
	(
		Message[ServiceExecute::niden];
		Throw[$Failed]
	)];
	
	Switch[input,
	{"CompoundID"},
	(
		newparams=newparams/.{Rule["CompoundID",a_]:>Rule["Compound",{"CompoundID",a}]} (*FALTAAA*)
	),{"Name"},
	(
		newparams=newparams/.{Rule["Name",a_]:>Rule["Compound",{"Name",a}]}
	),{"InChI"},
	(
		newparams=newparams/.{Rule["InChI",a_]:>Rule["Compound",{"InChI",a}]}
	),{"InChIKey"},
	(
		newparams=newparams/.{Rule["InChIKey",a_]:>Rule["Compound",{"InChIKey",a}]}
	),{"SMILES"},
	(
		newparams=newparams/.{Rule["SMILES",a_]:>Rule["Compound",{"SMILES",a}]}
	),{"Formula"},
	(
		newparams=newparams/.{Rule["Formula",a_]:>Rule["Compound",a]}
	),{},
	(
		Message[ServiceExecute::nparam,"CompoundID, Name, InChI, InChIKey or SMILES","PubChem"];
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,Method],
	(
		If[!StringMatchQ[Method/.newparams, "Similarity2DSearch"|"Similarity3DSearch"|"SubstructureSearch"|"SuperstructureSearch"|"FormulaSearch"],
		(	
			Message[ServiceExecute::nval,"Method","PubChem"];
			Throw[$Failed]
		)];
		search=(Method/.newparams)/.{"Similarity2DSearch"->"fastsimilarity_2d","Similarity3DSearch"->"fastsimilarity_3d","SubstructureSearch"->"fastsubstructure","SuperstructureSearch"->"fastsuperstructure","FormulaSearch"->"fastformula"};
		(*search parameters*)
		If[KeyExistsQ[newparams,#],
		(
			If[!MemberQ[{True,False},#/.newparams],
			(	
				Message[ServiceExecute::nval,#,"PubChem"];
				Throw[$Failed]
			)];
			params=Append[params,Rule[#,(#/.newparams)/.{True->"true",False->"false"}]]
		)]&/@searchparams[[1;;8]];
		If[KeyExistsQ[newparams,"Stereo"],
		(
			If[!StringMatchQ["Stereo"/.newparams, "ignore"|"exact"|"relative"|"nonconflicting",IgnoreCase->True],
			(	
				Message[ServiceExecute::nval,"Stereo","PubChem"];
				Throw[$Failed]
			)];
			params=Append[params,Rule["Stereo",("Stereo"/.newparams)]]
		)];
		If[KeyExistsQ[newparams,"Threshold"],
		(
			If[!IntegerQ["Threshold"/.newparams],
			(	
				Message[ServiceExecute::nval,"Threshold","PubChem"];
				Throw[$Failed]
			)];
			params=Append[params,Rule["Threshold",ToString[("Threshold"/.newparams)]]]
		)];
		
		If[MatchQ[search,"fastformula"],
		(
			If[KeyExistsQ[newparams,"Compound"],
			(
				If[!StringQ["Compound"/.newparams],
				(	
					Message[ServiceExecute::nval,input[[1]],"PubChem"];
					Throw[$Failed]
				)];
				namespace=search;
				identifiers="Compound"/.newparams;
			),
			(
				Message[ServiceExecute::nparam,input[[1]],"PubChem"];
				Throw[$Failed]
			)];
		),
		(
			If[KeyExistsQ[newparams,"Compound"],
			(
				compound="Compound"/.newparams;
				If[!MatchQ[compound,List[_String,_String|_Integer]],
				(	
					Message[ServiceExecute::nval,input[[1]],"PubChem"];
					Throw[$Failed]
				)];
				
				If[!StringMatchQ[compound[[1]], "CompoundID"|"SMILES"|"InChI"],
				(	
					Message[ServiceExecute::nval,input[[1]],"PubChem"];
					Throw[$Failed]
				)];
				namespace=StringJoin[search,"/",(compound[[1]]/.{"CompoundID"->"cid","SMILES"->"smiles","InChI"->"inchi"})];
				identifiers=ToString[compound[[2]]];
			),
			(
				Message[ServiceExecute::nparam,input[[1]],"PubChem"];
				Throw[$Failed]
			)];
		)]
	),
	(
		If[KeyExistsQ[newparams,"Compound"],
		(
			compound="Compound"/.newparams;
			If[!MatchQ[compound,List[_String,_String|_Integer|_Entity|List[__Entity]|List[__String]|List[__Integer]]],
			(	
				Message[ServiceExecute::nval,input[[1]],"PubChem"];
				Throw[$Failed]
			)];
			If[!((StringMatchQ[compound[[1]], "CompoundID"|"SMILES"|"InChIKey"|"Name"|"InChI"]&&MatchQ[compound,List[_String,_String|_Integer|List[__String]|List[__Integer]]])||(MatchQ[compound,List["Name",Entity["Chemical",_]]])||(MatchQ[compound,List["Name",List[Entity["Chemical", _] ..]]])),
			(	
				Message[ServiceExecute::nval,input[[1]],"PubChem"];
				Throw[$Failed]
			)];
			compound={compound[[1]]/.{"CompoundID"->"cid","SMILES"->"smiles","Name"->"name","InChI"->"inchi","InChIKey"->"inchikey"},compound[[2]]};
			If[MatchQ[compound,List["name",Entity["Chemical",_]]],
			(	
				namespace=compound[[1]];
				identifiers=ToString[compound[[2]]["Name"]]
			)];
			If[MatchQ[compound,List["name",List[Entity["Chemical",_]..]]],
			(	
				namespace=compound[[1]];
				identifiers=ToString[#["Name"]]&/@compound[[2]]
			)];
			If[MatchQ[compound,List["cid"|"smiles"|"inchikey"|"name"|"inchi",_String|_Integer]],
			(	
				namespace=compound[[1]];
				identifiers=ToString[compound[[2]]]
			)];
			If[MatchQ[compound,List["cid"|"smiles"|"inchikey",List[__String]|List[__Integer]]],
			(	
				namespace=compound[[1]];
				identifiers=StringJoin[Riffle[ToString/@compound[[2]],","]]
			)];
			If[MatchQ[compound,List["name"|"inchi",List[__String]|List[__Integer]]],
			(	
				namespace=compound[[1]];
				identifiers=ToString/@compound[[2]]
			)];
		),
		(
			Message[ServiceExecute::nparam,input[[1]],"PubChem"];
			Throw[$Failed]
		)];
	)];
	If[KeyExistsQ[newparams,"Property"],
	(
		If[!(StringQ["Property"/.newparams]||MatchQ["Property"/.newparams,List[__String]|All]),
		(	
			Message[ServiceExecute::nval,"Property","PubChem"];
			Throw[$Failed]
		)];
		If[!MatchQ["Property"/.newparams,All],
		(
			prop=Flatten[{"Property"/.newparams}];
			If[!(And@@StringMatchQ[prop, Alternatives@@allprops]),
			(	
				Message[ServiceExecute::nval,"Property","PubChem"];
				Throw[$Failed]
			)];
			prop2=StringJoin[Prepend[Riffle[prop,","],"property/"]]
		),
		(
			prop=allprops;
			prop2=StringJoin[Prepend[Riffle[allprops,","],"property/"]]
		)]
	),
	(
		prop=allprops;
		prop2=StringJoin[Prepend[Riffle[allprops,","],"property/"]]
	)];
	
	Switch[namespace,
	"fastsimilarity_2d/inchi"|"fastsimilarity_3d/inchi"|"fastsubstructure/inchi"|"fastsuperstructure/inchi",
	(
		rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompoundsPost",{"namespace" -> namespace, "identifiers" -> StringJoin["inchi=", identifiers, Sequence @@ (params /. {Rule[a_, b_] :> StringJoin["&", a, "=", b]})], "operation" -> prop2, "format" -> "JSON"}],"RawJSON"]]
	),"inchi",
	(
		If[MatchQ[compound,List["inchi",List[__]]],
		(	
			rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompoundsPost",{"namespace" -> namespace, "identifiers" -> StringJoin["inchi=", #, Sequence @@ (params /. {Rule[a_, b_] :> StringJoin["&", a, "=", b]})], "operation" -> prop2, "format" -> "JSON"}],"RawJSON"]]& /@ identifiers
		),
		(
			rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompoundsPost",{"namespace" -> namespace, "identifiers" -> StringJoin["inchi=", identifiers, Sequence @@ (params /. {Rule[a_, b_] :> StringJoin["&", a, "=", b]})], "operation" -> prop2, "format" -> "JSON"}],"RawJSON"]]
		)];
	),"name",
	(
		If[MatchQ[compound,List["name",List[__]]],
		(	
			rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompounds",Join[{"namespace" -> namespace, "identifiers" -> #, "operation" -> prop2, "format" -> "JSON"},params]],"RawJSON"]]&/@identifiers
		),
		(
			rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompounds",Join[{"namespace" -> namespace, "identifiers" -> identifiers, "operation" -> prop2, "format" -> "JSON"},params]],"RawJSON"]]
		)];
	),_,
	(
		rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompounds",Join[{"namespace" -> namespace, "identifiers" -> identifiers, "operation" -> prop2, "format" -> "JSON"},params]],"RawJSON"]]
	)];	
	If[MatchQ[compound,List["name"|"inchi",List[__]]],
	(
		If[MatchQ[#,$Failed],
   		(
    	  	Message[ServiceExecute::serrormsg,""];
    	   	Throw[$Failed]
 		)]&/@rawdata;
 		If[KeyExistsQ[#,"Fault"],
   		(
    	  	Message[ServiceExecute::serrormsg,"Message" /. ("Fault" /. #)];
    	   	Throw[$Failed]
 		)]&/@rawdata;
 	),
 	(
		If[MatchQ[rawdata,$Failed],
   		(
    	  	Message[ServiceExecute::serrormsg,""];
    	   	Throw[$Failed]
 		)];
 		If[KeyExistsQ[rawdata,"Fault"],
   		(
    	  	Message[ServiceExecute::serrormsg,"Message" /. ("Fault" /. rawdata)];
    	   	Throw[$Failed]
 		)];
 	)];
	rawdata = FixedPoint[Normal, rawdata];
 	rawdata = "Properties" /. ("PropertyTable" /. rawdata);
	withCamelTitles=Association @@@(rawdata/.{"CID"->"CompoundID"});
	withCamelTitles=Join[#, Association[Replace[DeleteCases[Sort[Append[prop, "CompoundID"]], Alternatives @@ Sort[Keys[#]]], {a_ :> Rule[a, Missing["NotAvailable"]]}, 1]]] & /@ withCamelTitles;
	Dataset[withCamelTitles][All,Flatten[{"CompoundID",prop}]]
]

pubchemcookeddata["CompoundImage", id_,args_]:=Block[{rawdata,newparams,input,compound,namespace,identifiers,type,size,params={},invalidParameters},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"CompoundID","Name","InChI","InChIKey","SMILES","ImageType","ImageSize"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"PubChem"]&/@invalidParameters;
		Throw[$Failed]
	)];
	input = Select[Keys[newparams],MemberQ[{"CompoundID","Name","InChI","InChIKey","SMILES"},#]&]; 
	If[Length[input]>1,
	(
		Message[ServiceExecute::niden];
		Throw[$Failed]
	)];
	
	Switch[input,
	{"CompoundID"},
	(
		newparams=newparams/.{Rule["CompoundID",a_]:>Rule["Compound",{"CompoundID",a}]} (*FALTAAA*)
	),{"Name"},
	(
		newparams=newparams/.{Rule["Name",a_]:>Rule["Compound",{"Name",a}]}
	),{"InChI"},
	(
		newparams=newparams/.{Rule["InChI",a_]:>Rule["Compound",{"InChI",a}]}
	),{"InChIKey"},
	(
		newparams=newparams/.{Rule["InChIKey",a_]:>Rule["Compound",{"InChIKey",a}]}
	),{"SMILES"},
	(
		newparams=newparams/.{Rule["SMILES",a_]:>Rule["Compound",{"SMILES",a}]}
	),{},
	(
		Message[ServiceExecute::nparam,"CompoundID, Name, InChI, InChIKey or SMILES","PubChem"];
		Throw[$Failed]
	)];
	
	If[KeyExistsQ[newparams,"Compound"],
	(
		compound="Compound"/.newparams;
		If[!MatchQ[compound,List[_String,_String|_Integer|_Entity]],
		(
			Message[ServiceExecute::nval,input[[1]],"PubChem"];
			Throw[$Failed]
		)];
		If[!((StringMatchQ[compound[[1]], "CompoundID"|"SMILES"|"Name"|"InChI"|"InChIKey"]&&MatchQ[compound,List[_String,_String|_Integer]])||(MatchQ[compound,List["Name",Entity["Chemical",_]]])),
		(	
			Message[ServiceExecute::nval,input[[1]],"PubChem"];
			Throw[$Failed]
		)];
		compound={compound[[1]]/.{"CompoundID"->"cid","SMILES"->"smiles","Name"->"name","InChI"->"inchi","InChIKey"->"inchikey"},compound[[2]]};
		If[MatchQ[compound,List["name",Entity["Chemical",_]]],
		(	
			namespace=compound[[1]];
			identifiers=ToString[compound[[2]]["Name"]]
		)];
		If[MatchQ[compound,List[_String,_String|_Integer]],
		(	
			namespace=compound[[1]];
			identifiers=ToString[compound[[2]]]
		)];
	),
	(
		Message[ServiceExecute::nparam,input[[1]],"PubChem"];
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"ImageType"],
	(
		If[!MemberQ[{"2D","3D"},"ImageType"/.newparams],
		(
			Message[ServiceExecute::nval,"ImageType","PubChem"];
			Throw[$Failed]
		)];
		type=("ImageType"/.newparams)/.{"2D"->"2d","3D"->"3d"};
	),
	(
		type="2d"
	)];
	If[KeyExistsQ[newparams,"ImageSize"],
	(
		If[!MemberQ[{"Small","Large"},ToString["ImageSize"/.newparams]],
		(
			Message[ServiceExecute::nval,"ImageSize","PubChem"];
			Throw[$Failed]
		)];
		size=ToString["ImageSize"/.newparams];
	),
	(
		size="Large"
	)];
	
	If[MatchQ[namespace,"inchi"],
	(
		rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompoundImagePost",{"namespace" -> namespace, "identifiers" -> StringJoin["inchi=", identifiers, Sequence @@ ({"record_type"->type,"image_size"->size} /. {Rule[a_, b_] :> StringJoin["&", a, "=", b]})]}],"PNG"]]
	),
	(
		rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompoundImage",{"namespace" -> namespace, "identifiers" -> identifiers,"record_type"->type,"image_size"->size}],"PNG"]]
	)];
	If[MatchQ[rawdata,$Failed],
   	(
      	Message[ServiceExecute::serrormsg,""];
       	Throw[$Failed]
 	)];
 	rawdata	
]

pubchemcookeddata["CompoundFullRecords", id_,args_]:=Block[{rawdata,newparams,input,compound,namespace,identifiers,prop,params={},invalidParameters,withCamelTitles,search},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[Join[{"CompoundID","Name","InChI","InChIKey","SMILES","Formula",Method,"RecordType"},searchparams],#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"PubChem"]&/@invalidParameters;
		Throw[$Failed]
	)];
	input = Select[Keys[newparams],MemberQ[{"CompoundID","Name","InChI","InChIKey","SMILES","Formula"},#]&]; 
	If[Length[input]>1,
	(
		Message[ServiceExecute::niden];
		Throw[$Failed]
	)];
	
	Switch[input,
	{"CompoundID"},
	(
		newparams=newparams/.{Rule["CompoundID",a_]:>Rule["Compound",{"CompoundID",a}]} (*FALTAAA*)
	),{"Name"},
	(
		newparams=newparams/.{Rule["Name",a_]:>Rule["Compound",{"Name",a}]}
	),{"InChI"},
	(
		newparams=newparams/.{Rule["InChI",a_]:>Rule["Compound",{"InChI",a}]}
	),{"InChIKey"},
	(
		newparams=newparams/.{Rule["InChIKey",a_]:>Rule["Compound",{"InChIKey",a}]}
	),{"SMILES"},
	(
		newparams=newparams/.{Rule["SMILES",a_]:>Rule["Compound",{"SMILES",a}]}
	),{"Formula"},
	(
		newparams=newparams/.{Rule["Formula",a_]:>Rule["Compound",a]}
	),{},
	(
		Message[ServiceExecute::nparam,"CompoundID, Name, InChI, InChIKey or SMILES","PubChem"];
		Throw[$Failed]
	)];
	
	If[KeyExistsQ[newparams,Method],
	(
		If[!StringMatchQ[Method/.newparams, "Similarity2DSearch"|"Similarity3DSearch"|"SubstructureSearch"|"SuperstructureSearch"|"FormulaSearch"],
		(	
			Message[ServiceExecute::nval,"Method","PubChem"];
			Throw[$Failed]
		)];
		search=(Method/.newparams)/.{"Similarity2DSearch"->"fastsimilarity_2d","Similarity3DSearch"->"fastsimilarity_3d","SubstructureSearch"->"fastsubstructure","SuperstructureSearch"->"fastsuperstructure","FormulaSearch"->"fastformula"};
		(*search parameters*)
		If[KeyExistsQ[newparams,#],
		(
			If[!MemberQ[{True,False},#/.newparams],
			(	
				Message[ServiceExecute::nval,#,"PubChem"];
				Throw[$Failed]
			)];
			params=Append[params,Rule[#,(#/.newparams)/.{True->"true",False->"false"}]]
		)]&/@searchparams[[1;;8]];
		If[KeyExistsQ[newparams,"Stereo"],
		(
			If[!StringMatchQ["Stereo"/.newparams, "ignore"|"exact"|"relative"|"nonconflicting",IgnoreCase->True],
			(	
				Message[ServiceExecute::nval,"Stereo","PubChem"];
				Throw[$Failed]
			)];
			params=Append[params,Rule["Stereo",("Stereo"/.newparams)]]
		)];
		If[KeyExistsQ[newparams,"Threshold"],
		(
			If[!IntegerQ["Threshold"/.newparams],
			(	
				Message[ServiceExecute::nval,"Threshold","PubChem"];
				Throw[$Failed]
			)];
			params=Append[params,Rule["Threshold",ToString[("Threshold"/.newparams)]]]
		)];
		
		If[MatchQ[search,"fastformula"],
		(
			If[KeyExistsQ[newparams,"Compound"],
			(
				If[!StringQ["Compound"/.newparams],
				(	
					Message[ServiceExecute::nval,input[[1]],"PubChem"];
					Throw[$Failed]
				)];
				namespace=search;
				identifiers="Compound"/.newparams;
			),
			(
				Message[ServiceExecute::nparam,input[[1]],"PubChem"];
				Throw[$Failed]
			)];
		),
		(
			If[KeyExistsQ[newparams,"Compound"],
			(
				compound="Compound"/.newparams;
				If[!MatchQ[compound,List[_String,_String|_Integer]],
				(	
					Message[ServiceExecute::nval,input[[1]],"PubChem"];
					Throw[$Failed]
				)];
				
				If[!StringMatchQ[compound[[1]], "CompoundID"|"SMILES"|"InChI"],
				(	
					Message[ServiceExecute::nval,input[[1]],"PubChem"];
					Throw[$Failed]
				)];
				namespace=StringJoin[search,"/",(compound[[1]]/.{"CompoundID"->"cid","SMILES"->"smiles","InChI"->"inchi"})];
				identifiers=ToString[compound[[2]]];
			),
			(
				Message[ServiceExecute::nparam,input[[1]],"PubChem"];
				Throw[$Failed]
			)];
		)]
	),
	(
		If[KeyExistsQ[newparams,"Compound"],
		(
			compound="Compound"/.newparams;
			If[!MatchQ[compound,List[_String,_String|_Integer|_Entity|List[__Entity]|List[__String]|List[__Integer]]],
			(	
				Message[ServiceExecute::nval,input[[1]],"PubChem"];
				Throw[$Failed]
			)];
			If[!((StringMatchQ[compound[[1]], "CompoundID"|"SMILES"|"InChIKey"|"Name"|"InChI"]&&MatchQ[compound,List[_String,_String|_Integer|List[__String]|List[__Integer]]])||(MatchQ[compound,List["Name",Entity["Chemical",_]]])||(MatchQ[compound,List["Name",List[Entity["Chemical", _] ..]]])),
			(	
				Message[ServiceExecute::nval,input[[1]],"PubChem"];
				Throw[$Failed]
			)];
			compound={compound[[1]]/.{"CompoundID"->"cid","SMILES"->"smiles","Name"->"name","InChI"->"inchi","InChIKey"->"inchikey"},compound[[2]]};
			If[MatchQ[compound,List["name",Entity["Chemical",_]]],
			(	
				namespace=compound[[1]];
				identifiers=ToString[compound[[2]]["Name"]]
			)];
			If[MatchQ[compound,List["name",List[Entity["Chemical",_]..]]],
			(	
				namespace=compound[[1]];
				identifiers=ToString[#["Name"]]&/@compound[[2]]
			)];
			If[MatchQ[compound,List["cid"|"smiles"|"inchikey"|"name"|"inchi",_String|_Integer]],
			(	
				namespace=compound[[1]];
				identifiers=ToString[compound[[2]]]
			)];
			If[MatchQ[compound,List["cid"|"smiles"|"inchikey",List[__String]|List[__Integer]]],
			(	
				namespace=compound[[1]];
				identifiers=StringJoin[Riffle[ToString/@compound[[2]],","]]
			)];
			If[MatchQ[compound,List["name"|"inchi",List[__String]|List[__Integer]]],
			(	
				namespace=compound[[1]];
				identifiers=ToString/@compound[[2]]
			)];
		),
		(
			Message[ServiceExecute::nparam,input[[1]],"PubChem"];
			Throw[$Failed]
		)];
	)];
	If[KeyExistsQ[newparams,"RecordType"],
	(
		If[!MemberQ[{"2D","3D"},"RecordType"/.newparams],
		(
			Message[ServiceExecute::nval,"RecordType","PubChem"];
			Throw[$Failed]
		)];
		params=Append[params,Rule["record_type",("RecordType"/.newparams)/.{"2D"->"2d","3D"->"3d"}]];
	),
	(
		params=Append[params,Rule["record_type","2d"]]
	)];
	
	Switch[namespace,
	"fastsimilarity_2d/inchi"|"fastsimilarity_3d/inchi"|"fastsubstructure/inchi"|"fastsuperstructure/inchi",
	(
		rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompoundsPost",{"namespace" -> namespace, "identifiers" -> StringJoin["inchi=", identifiers, Sequence @@ (params /. {Rule[a_, b_] :> StringJoin["&", a, "=", b]})], "operation" -> "record", "format" -> "JSON"}],"RawJSON"]]
	),"inchi",
	(
		If[MatchQ[compound,List["inchi",List[__]]],
		(	
			rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompoundsPost",{"namespace" -> namespace, "identifiers" -> StringJoin["inchi=", #, Sequence @@ (params /. {Rule[a_, b_] :> StringJoin["&", a, "=", b]})], "operation" -> "record", "format" -> "JSON"}],"RawJSON"]]& /@ identifiers
		),
		(
			rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompoundsPost",{"namespace" -> namespace, "identifiers" -> StringJoin["inchi=", identifiers, Sequence @@ (params /. {Rule[a_, b_] :> StringJoin["&", a, "=", b]})], "operation" -> "record", "format" -> "JSON"}],"RawJSON"]]
		)];
	),"name",
	(
		If[MatchQ[compound,List["name",List[__]]],
		(	
			rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompounds",Join[{"namespace" -> namespace, "identifiers" -> #, "operation" -> "record", "format" -> "JSON"},params]],"RawJSON"]]&/@identifiers
		),
		(
			rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompounds",Join[{"namespace" -> namespace, "identifiers" -> identifiers, "operation" -> "record", "format" -> "JSON"},params]],"RawJSON"]]
		)];
	),_,
	(
		rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompounds",Join[{"namespace" -> namespace, "identifiers" -> identifiers, "operation" -> "record", "format" -> "JSON"},params]],"RawJSON"]]
	)];	
	If[MatchQ[compound,List["name"|"inchi",List[__]]],
	(
		If[MatchQ[#,$Failed],
   		(
    	  	Message[ServiceExecute::serrormsg,""];
    	   	Throw[$Failed]
 		)]&/@rawdata;
 		If[KeyExistsQ[#,"Fault"],
   		(
    	  	Message[ServiceExecute::serrormsg,"Message" /. ("Fault" /. #)];
    	   	Throw[$Failed]
 		)]&/@rawdata;
 	),
 	(
		If[MatchQ[rawdata,$Failed],
   		(
    	  	Message[ServiceExecute::serrormsg,""];
    	   	Throw[$Failed]
 		)];
 		If[KeyExistsQ[rawdata,"Fault"],
   		(
    	  	Message[ServiceExecute::serrormsg,"Message" /. ("Fault" /. rawdata)];
    	   	Throw[$Failed]
 		)];
 	)];
	rawdata = FixedPoint[Normal, rawdata];
 	rawdata = ("PC_Compounds" /. rawdata)/. {Rule["id", List[Rule["id", a_]]] :> Rule["id", a], Rule["value", List[Rule[_, c_]]] :> Rule["value", c], Rule["conformers", List[d_List]] :> Rule["conformers", d]} /. {Rule["coords", List[d_List]] :> Rule["coords", d]};
	withCamelTitles=Replace[rawdata, Rule[a_, b_] :> Rule[camelCase[a], b], Infinity] /. {"Id" -> "ID", "Cid" -> "CompoundID", "Aid" -> "AssayID", "Aid1" -> "AssayID1", "Aid2" -> "AssayID2", "Props" -> "Properties", "Urn" -> "URN"};
	Dataset[Replace[Association@@@withCamelTitles, r : {__Rule} :> Association[r], -1]]
]

pubchemcookeddata["CompoundAssaySummary", id_,args_]:=Block[{rawdata,newparams,input,compound,namespace,identifiers,prop,params={},invalidParameters,withCamelTitles,search,col,row},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[Join[{"CompoundID","Name","InChI","InChIKey","SMILES","Formula",Method},searchparams],#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"PubChem"]&/@invalidParameters;
		Throw[$Failed]
	)];
	input = Select[Keys[newparams],MemberQ[{"CompoundID","Name","InChI","InChIKey","SMILES","Formula"},#]&]; 
	If[Length[input]>1,
	(
		Message[ServiceExecute::niden];
		Throw[$Failed]
	)];
	
	Switch[input,
	{"CompoundID"},
	(
		newparams=newparams/.{Rule["CompoundID",a_]:>Rule["Compound",{"CompoundID",a}]} (*FALTAAA*)
	),{"Name"},
	(
		newparams=newparams/.{Rule["Name",a_]:>Rule["Compound",{"Name",a}]}
	),{"InChI"},
	(
		newparams=newparams/.{Rule["InChI",a_]:>Rule["Compound",{"InChI",a}]}
	),{"InChIKey"},
	(
		newparams=newparams/.{Rule["InChIKey",a_]:>Rule["Compound",{"InChIKey",a}]}
	),{"SMILES"},
	(
		newparams=newparams/.{Rule["SMILES",a_]:>Rule["Compound",{"SMILES",a}]}
	),{"Formula"},
	(
		newparams=newparams/.{Rule["Formula",a_]:>Rule["Compound",a]}
	),{},
	(
		Message[ServiceExecute::nparam,"CompoundID, Name, InChI, InChIKey or SMILES","PubChem"];
		Throw[$Failed]
	)];
		
	If[KeyExistsQ[newparams,Method],
	(
		If[!StringMatchQ[Method/.newparams, "Similarity2DSearch"|"Similarity3DSearch"|"SubstructureSearch"|"SuperstructureSearch"|"FormulaSearch"],
		(	
			Message[ServiceExecute::nval,"Method","PubChem"];
			Throw[$Failed]
		)];
		search=(Method/.newparams)/.{"Similarity2DSearch"->"fastsimilarity_2d","Similarity3DSearch"->"fastsimilarity_3d","SubstructureSearch"->"fastsubstructure","SuperstructureSearch"->"fastsuperstructure","FormulaSearch"->"fastformula"};
		(*search parameters*)
		If[KeyExistsQ[newparams,#],
		(
			If[!MemberQ[{True,False},#/.newparams],
			(	
				Message[ServiceExecute::nval,#,"PubChem"];
				Throw[$Failed]
			)];
			params=Append[params,Rule[#,(#/.newparams)/.{True->"true",False->"false"}]]
		)]&/@searchparams[[1;;8]];
		If[KeyExistsQ[newparams,"Stereo"],
		(
			If[!StringMatchQ["Stereo"/.newparams, "ignore"|"exact"|"relative"|"nonconflicting",IgnoreCase->True],
			(	
				Message[ServiceExecute::nval,"Stereo","PubChem"];
				Throw[$Failed]
			)];
			params=Append[params,Rule["Stereo",("Stereo"/.newparams)]]
		)];
		If[KeyExistsQ[newparams,"Threshold"],
		(
			If[!IntegerQ["Threshold"/.newparams],
			(	
				Message[ServiceExecute::nval,"Threshold","PubChem"];
				Throw[$Failed]
			)];
			params=Append[params,Rule["Threshold",ToString[("Threshold"/.newparams)]]]
		)];
		
		If[MatchQ[search,"fastformula"],
		(
			If[KeyExistsQ[newparams,"Compound"],
			(
				If[!StringQ["Compound"/.newparams],
				(	
					Message[ServiceExecute::nval,input[[1]],"PubChem"];
					Throw[$Failed]
				)];
				namespace=search;
				identifiers="Compound"/.newparams;
			),
			(
				Message[ServiceExecute::nparam,input[[1]],"PubChem"];
				Throw[$Failed]
			)];
		),
		(
			If[KeyExistsQ[newparams,"Compound"],
			(
				compound="Compound"/.newparams;
				If[!MatchQ[compound,List[_String,_String|_Integer]],
				(	
					Message[ServiceExecute::nval,input[[1]],"PubChem"];
					Throw[$Failed]
				)];
				
				If[!StringMatchQ[compound[[1]], "CompoundID"|"SMILES"|"InChI"],
				(	
					Message[ServiceExecute::nval,input[[1]],"PubChem"];
					Throw[$Failed]
				)];
				namespace=StringJoin[search,"/",(compound[[1]]/.{"CompoundID"->"cid","SMILES"->"smiles","InChI"->"inchi"})];
				identifiers=ToString[compound[[2]]];
			),
			(
				Message[ServiceExecute::nparam,input[[1]],"PubChem"];
				Throw[$Failed]
			)];
		)]
	),
	(
		If[KeyExistsQ[newparams,"Compound"],
		(
			compound="Compound"/.newparams;
			If[!MatchQ[compound,List[_String,_String|_Integer|_Entity|List[__String]|List[__Integer]]],
			(	
				Message[ServiceExecute::nval,input[[1]],"PubChem"];
				Throw[$Failed]
			)];
			If[!((StringMatchQ[compound[[1]], "CompoundID"|"SMILES"|"InChIKey"]&&MatchQ[compound,List[_String,_String|_Integer|List[__String]|List[__Integer]]])||(StringMatchQ[compound[[1]],"Name"|"InChI"]&&MatchQ[compound,List[_String,_String]])||(MatchQ[compound,List["Name",Entity["Chemical",_]]])),
			(	
				Message[ServiceExecute::nval,input[[1]],"PubChem"];
				Throw[$Failed]
			)];
			compound={compound[[1]]/.{"CompoundID"->"cid","SMILES"->"smiles","Name"->"name","InChI"->"inchi","InChIKey"->"inchikey"},compound[[2]]};
			If[MatchQ[compound,List["name",Entity["Chemical",_]]],
			(	
				namespace=compound[[1]];
				identifiers=ToString[compound[[2]]["Name"]]
			)];
			If[MatchQ[compound,List[_String,_String|_Integer]],
			(	
				namespace=compound[[1]];
				identifiers=ToString[compound[[2]]]
			)];
			If[MatchQ[compound,List[_String,List[__String]|List[__Integer]]],
			(	
				namespace=compound[[1]];
				identifiers=StringJoin[Riffle[ToString/@compound[[2]],","]]
			)];
		),
		(
			Message[ServiceExecute::nparam,input[[1]],"PubChem"];
			Throw[$Failed]
		)];
	)];
	If[MatchQ[namespace,"inchi"|"fastsimilarity_2d/inchi"|"fastsimilarity_3d/inchi"|"fastsubstructure/inchi"|"fastsuperstructure/inchi"],
	(
		rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompoundsPost",{"namespace" -> namespace, "identifiers" -> StringJoin["inchi=", identifiers, Sequence @@ (params /. {Rule[a_, b_] :> StringJoin["&", a, "=", b]})], "operation" -> "assaysummary", "format" -> "JSON"}],"RawJSON"]]
	),
	(
		rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawCompounds",Join[{"namespace" -> namespace, "identifiers" -> identifiers, "operation" -> "assaysummary", "format" -> "JSON"},params]],"RawJSON"]]
	)];
	If[MatchQ[rawdata,$Failed],
   	(
      	Message[ServiceExecute::serrormsg,""];
       	Throw[$Failed]
 	)];
 	If[KeyExistsQ[rawdata,"Fault"],
   	(
      	Message[ServiceExecute::serrormsg,"Message" /. ("Fault" /. rawdata)];
       	Throw[$Failed]
 	)];
 	rawdata = FixedPoint[Normal, rawdata];
 	col = "Column" /. ("Columns" /. ("Table" /. rawdata));
 	row = ("Cell" /. ("Row" /. ("Table" /. rawdata))) /. {"" -> Missing["NotAvailable"]};
 	Dataset[Association @@@ Reverse[Inner[Rule, row, col, List], 3]]
]

pubchemcookeddata["CompoundSDF", id_,args_]:=Block[{rawdata,newparams,input,compound,mol,namespace,identifiers,type,params={},invalidParameters,elem},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"CompoundID","Name","InChI","InChIKey","SMILES"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"PubChem"]&/@invalidParameters;
		Throw[$Failed]
	)];
	input = Select[Keys[newparams],MemberQ[{"CompoundID","Name","InChI","InChIKey","SMILES"},#]&]; 
	If[Length[input]>1,
	(
		Message[ServiceExecute::niden];
		Throw[$Failed]
	)];
	
	Switch[input,
	{"CompoundID"},
	(
		newparams=newparams/.{Rule["CompoundID",a_]:>Rule["Compound",{"CompoundID",a}]}
	),{"Name"},
	(
		newparams=newparams/.{Rule["Name",a_]:>Rule["Compound",{"Name",a}]}
	),{"InChI"},
	(
		newparams=newparams/.{Rule["InChI",a_]:>Rule["Compound",{"InChI",a}]}
	),{"InChIKey"},
	(
		newparams=newparams/.{Rule["InChIKey",a_]:>Rule["Compound",{"InChIKey",a}]}
	),{"SMILES"},
	(
		newparams=newparams/.{Rule["SMILES",a_]:>Rule["Compound",{"SMILES",a}]}
	),{},
	(
		Message[ServiceExecute::nparam,"CompoundID, Name, InChI, InChIKey or SMILES","PubChem"];
		Throw[$Failed]
	)];
	
	If[KeyExistsQ[newparams,"Compound"],
	(
		compound="Compound"/.newparams;
		If[!MatchQ[compound,List[_String,_String|_Integer|_Entity]],
		(
			Message[ServiceExecute::nval,input[[1]],"PubChem"];
			Throw[$Failed]
		)];
		If[!((StringMatchQ[compound[[1]], "CompoundID"|"SMILES"|"Name"|"InChI"|"InChIKey"]&&MatchQ[compound,List[_String,_String|_Integer]])||(MatchQ[compound,List["Name",Entity["Chemical",_]]])),
		(	
			Message[ServiceExecute::nval,input[[1]],"PubChem"];
			Throw[$Failed]
		)];
		compound={compound[[1]]/.{"CompoundID"->"cid","SMILES"->"smiles","Name"->"name","InChI"->"inchi","InChIKey"->"inchikey"},compound[[2]]};
		If[MatchQ[compound,List["name",Entity["Chemical",_]]],
		(	
			namespace=compound[[1]];
			identifiers=ToString[compound[[2]]["Name"]]
		)];
		If[MatchQ[compound,List[_String,_String|_Integer]],
		(	
			namespace=compound[[1]];
			identifiers=ToString[compound[[2]]]
		)];
	),
	(
		Message[ServiceExecute::nparam,input[[1]],"PubChem"];
		Throw[$Failed]
	)];
	
	If[MatchQ[namespace,"inchi"],
	(
		rawdata = ServiceExecute["PubChem","RawCompoundSDFPost",{"namespace" -> namespace, "identifiers" -> StringJoin["inchi=", identifiers]}]
	),
	(
		rawdata = ServiceExecute["PubChem","RawCompoundSDF",{"namespace" -> namespace, "identifiers" -> identifiers}]
	)];
	elem = ImportString[rawdata, {"SDF", "Elements"}];
	(*mol = StringSplit[StringReplace[rawdata, "$$$$\n" -> "$$$$SPLIT"], "SPLIT"];*)
	rawdata = Append[Rule[#, ImportString[rawdata, {"SDF", #}]] & /@ elem, Rule["MOL", {rawdata}]];

	If[MatchQ[#,Rule[_,$Failed]],
   	(
      	Message[ServiceExecute::serrormsg,""];
       	Throw[$Failed]
 	)]&/@rawdata;
 	Dataset[Association @@ rawdata]	
]


pubchemcookeddata[req:"SubstanceSynonyms"|"SubstanceSID"|"SubstanceAID"|"SubstanceCID", id_,args_]:=Block[{rawdata,newparams,input,substance,namespace,identifiers,operation,params={},invalidParameters,withCamelTitles},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"SubstanceID","Name","AIDType","SIDType","CIDType"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"PubChem"]&/@invalidParameters;
		Throw[$Failed]
	)];
	input = Select[Keys[newparams],MemberQ[{"SubstanceID","Name"},#]&]; 
	If[Length[input]>1,
	(
		Message[ServiceExecute::niden];
		Throw[$Failed]
	)];
	
	Switch[input,
	{"SubstanceID"},
	(
		newparams=newparams/.{Rule["SubstanceID",a_]:>Rule["Substance",{"SubstanceID",a}]}
	),{"Name"},
	(
		newparams=newparams/.{Rule["Name",a_]:>Rule["Substance",{"Name",a}]}
	),{},
	(
		Message[ServiceExecute::nparam,"SubstanceID or Name","PubChem"];
		Throw[$Failed]
	)];
	
	If[KeyExistsQ[newparams,"Substance"],
	(
		substance="Substance"/.newparams;		
		If[!MatchQ[substance,List[_String,_String|_Integer|_Entity|List[__Entity]|List[__String]|List[__Integer]]],
		(	
			Message[ServiceExecute::nval,input[[1]],"PubChem"];
			Throw[$Failed]
		)];
		If[!((StringMatchQ[substance[[1]], "SubstanceID"|"Name"]&&MatchQ[substance,List[_String,_String|_Integer|List[__String]|List[__Integer]]])||(MatchQ[substance,List["Name",Entity["Chemical",_]]])||(MatchQ[substance,List["Name",List[Entity["Chemical", _] ..]]])),
		(	
			Message[ServiceExecute::nval,input[[1]],"PubChem"];
			Throw[$Failed]
		)];
		substance={substance[[1]]/.{"SubstanceID"->"sid","Name"->"name"},substance[[2]]};
		If[MatchQ[substance,List[_String,_String|_Integer]],
		(	
			namespace=substance[[1]];
			identifiers=ToString[substance[[2]]]
		)];
		If[MatchQ[substance,List["sid",List[__String]|List[__Integer]]],
		(	
			namespace=substance[[1]];
			identifiers=StringJoin[Riffle[ToString/@substance[[2]],","]]
		)];
		If[MatchQ[substance,List["name",List[__String]|List[__Integer]]],
		(	
			namespace=substance[[1]];
			identifiers=ToString/@substance[[2]]
		)];
		If[MatchQ[substance,List["name",Entity["Chemical",_]]],
		(	
			namespace=substance[[1]];
			identifiers=ToString[substance[[2]]["Name"]]
		)];
		If[MatchQ[substance,List["name",List[Entity["Chemical",_]..]]],
		(	
			namespace=substance[[1]];
			identifiers=ToString[#["Name"]]&/@substance[[2]]
		)];
	),
	(
		Message[ServiceExecute::nparam,input[[1]],"PubChem"];
		Throw[$Failed]
	)];
	
	Switch[req,
		"SubstanceSynonyms",operation="synonyms",
		"SubstanceCID",
			operation="cids";
			If[KeyExistsQ[newparams,"CIDType"],
			(
				If[!StringMatchQ[ToString["CIDType"/.newparams], "All"|"Standardized"|"Component"],
				(	
					Message[ServiceExecute::nval,"CIDType","PubChem"];
					Throw[$Failed]
				)];
				params=Append[params,Rule["cids_type",ToString["CIDType"/.newparams]/.{"All"->"all","Standardized"->"standardized","Component"->"component"}]]
			)];,
		"SubstanceAID",
			operation="aids";
			If[KeyExistsQ[newparams,"AIDType"],
			(
				If[!StringMatchQ[ToString["AIDType"/.newparams], "All"|"Active"|"Inactive"],
				(	
					Message[ServiceExecute::nval,"AIDType","PubChem"];
					Throw[$Failed]
				)];
				params=Append[params,Rule["aids_type",ToString["AIDType"/.newparams]/.{"All"->"all","Active"->"active","Inactive"->"inactive"}]]
			)];,
		"SubstanceSID",
			operation="sids";
			If[KeyExistsQ[newparams,"SIDType"],
			(
				If[!StringMatchQ[ToString["SIDType"/.newparams], (Alternatives @@ sidtype[[All, 1]])],
				(	
					Message[ServiceExecute::nval,"SIDType","PubChem"];
					Throw[$Failed]
				)];
				params=Append[params,Rule["sids_type",ToString["SIDType"/.newparams]/.sidtype]]
			)];
	];
	
	If[MatchQ[substance,List["name",List[__]]],
	(	
		rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawSubstances",Join[{"namespace" -> namespace, "identifiers" -> #, "operation" -> operation, "format" -> "JSON"},params]],"RawJSON"]]&/@identifiers
	),
	(
		rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawSubstances",Join[{"namespace" -> namespace, "identifiers" -> identifiers, "operation" -> operation, "format" -> "JSON"},params]],"RawJSON"]]
	)];
	If[MatchQ[substance,List["name",List[__]]],
	(
		If[MatchQ[#,$Failed],
   		(
    	  	Message[ServiceExecute::serrormsg,""];
    	   	Throw[$Failed]
 		)]&/@rawdata;
 		If[KeyExistsQ[#,"Fault"],
   		(
    	  	Message[ServiceExecute::serrormsg,"Message" /. ("Fault" /. #)];
    	   	Throw[$Failed]
 		)]&/@rawdata;
 	),
 	(
		If[MatchQ[rawdata,$Failed],
   		(
    	  	Message[ServiceExecute::serrormsg,""];
    	   	Throw[$Failed]
 		)];
 		If[KeyExistsQ[rawdata,"Fault"],
   		(
    	  	Message[ServiceExecute::serrormsg,"Message" /. ("Fault" /. rawdata)];
    	   	Throw[$Failed]
 		)];
 	)];
	rawdata = FixedPoint[Normal, rawdata];
 	Switch[req,
		"SubstanceSynonyms"|"SubstanceCID"|"SubstanceAID",rawdata = "Information" /. ("InformationList" /. rawdata),
		"SubstanceSID",rawdata = "IdentifierList" /. rawdata
	];
 	withCamelTitles=rawdata/.{"SID"->"SubstanceID","CID"->"CompoundID","AID"->"AssayID"};
 	Switch[req,
		"SubstanceSynonyms"|"SubstanceCID"|"SubstanceAID",Dataset[Association @@@ withCamelTitles],
		"SubstanceSID",
		If[MatchQ[substance,List["name",List[__]]],
		(
			Dataset[Association /@ withCamelTitles]
 		),
		(
			Dataset[Association @@ withCamelTitles]
		)]
	]
	
]

pubchemcookeddata["SubstanceImage", id_,args_]:=Block[{rawdata,newparams,input,substance,namespace,identifiers,size,params={},invalidParameters},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"SubstanceID","Name","ImageSize"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"PubChem"]&/@invalidParameters;
		Throw[$Failed]
	)];	
	input = Select[Keys[newparams],MemberQ[{"SubstanceID","Name"},#]&]; 
	If[Length[input]>1,
	(
		Message[ServiceExecute::niden];
		Throw[$Failed]
	)];
	
	Switch[input,
	{"SubstanceID"},
	(
		newparams=newparams/.{Rule["SubstanceID",a_]:>Rule["Substance",{"SubstanceID",a}]}
	),{"Name"},
	(
		newparams=newparams/.{Rule["Name",a_]:>Rule["Substance",{"Name",a}]}
	),{},
	(
		Message[ServiceExecute::nparam,"SubstanceID or Name","PubChem"];
		Throw[$Failed]
	)];
	
	If[KeyExistsQ[newparams,"Substance"],
	(
		substance="Substance"/.newparams;
		If[!MatchQ[substance,List["SubstanceID",_String|_Integer]|List["Name",_String|Entity["Chemical",_]]],
		(	
			Message[ServiceExecute::nval,input[[1]],"PubChem"];
			Throw[$Failed]
		)];
		namespace=substance[[1]]/.{"SubstanceID"->"sid","Name"->"name"};
		If[!MatchQ[substance,List["name",Entity["Chemical",_]]],
		(
			identifiers=ToString[substance[[2]]]
		),
		(
			identifiers=ToString[substance[[2]]["Name"]]
		)]
	),
	(
		Message[ServiceExecute::nparam,input[[1]],"PubChem"];
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"ImageSize"],
	(
		If[!MemberQ[{"Small","Large"},ToString["ImageSize"/.newparams]],
		(
			Message[ServiceExecute::nval,"ImageSize","PubChem"];
			Throw[$Failed]
		)];
		size=ToString["ImageSize"/.newparams];
	),
	(
		size="Large"
	)];
	rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawSubstanceImage",{"namespace" -> namespace, "identifiers" -> identifiers,"image_size"->size}],"PNG"]];
	If[MatchQ[rawdata,$Failed],
   	(
      	Message[ServiceExecute::serrormsg,""];
       	Throw[$Failed]
 	)];
 	rawdata	
]

pubchemcookeddata["SubstanceFullRecords", id_,args_]:=Block[{rawdata,newparams,input,substance,namespace,identifiers,prop,params={},invalidParameters,withCamelTitles},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"SubstanceID","Name"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"PubChem"]&/@invalidParameters;
		Throw[$Failed]
	)];
	input = Select[Keys[newparams],MemberQ[{"SubstanceID","Name"},#]&]; 
	If[Length[input]>1,
	(
		Message[ServiceExecute::niden];
		Throw[$Failed]
	)];
	
	Switch[input,
	{"SubstanceID"},
	(
		newparams=newparams/.{Rule["SubstanceID",a_]:>Rule["Substance",{"SubstanceID",a}]}
	),{"Name"},
	(
		newparams=newparams/.{Rule["Name",a_]:>Rule["Substance",{"Name",a}]}
	),{},
	(
		Message[ServiceExecute::nparam,"SubstanceID or Name","PubChem"];
		Throw[$Failed]
	)];
	
	If[KeyExistsQ[newparams,"Substance"],
	(
		substance="Substance"/.newparams;		
		If[!MatchQ[substance,List[_String,_String|_Integer|_Entity|List[__Entity]|List[__String]|List[__Integer]]],
		(	
			Message[ServiceExecute::nval,input[[1]],"PubChem"];
			Throw[$Failed]
		)];
		If[!((StringMatchQ[substance[[1]], "SubstanceID"|"Name"]&&MatchQ[substance,List[_String,_String|_Integer|List[__String]|List[__Integer]]])||(MatchQ[substance,List["Name",Entity["Chemical",_]]])||(MatchQ[substance,List["Name",List[Entity["Chemical", _] ..]]])),
		(	
			Message[ServiceExecute::nval,input[[1]],"PubChem"];
			Throw[$Failed]
		)];
		substance={substance[[1]]/.{"SubstanceID"->"sid","Name"->"name"},substance[[2]]};
		If[MatchQ[substance,List[_String,_String|_Integer]],
		(	
			namespace=substance[[1]];
			identifiers=ToString[substance[[2]]]
		)];
		If[MatchQ[substance,List["sid",List[__String]|List[__Integer]]],
		(	
			namespace=substance[[1]];
			identifiers=StringJoin[Riffle[ToString/@substance[[2]],","]]
		)];
		If[MatchQ[substance,List["name",List[__String]|List[__Integer]]],
		(	
			namespace=substance[[1]];
			identifiers=ToString/@substance[[2]]
		)];
		If[MatchQ[substance,List["name",Entity["Chemical",_]]],
		(	
			namespace=substance[[1]];
			identifiers=ToString[substance[[2]]["Name"]]
		)];
		If[MatchQ[substance,List["name",List[Entity["Chemical",_]..]]],
		(	
			namespace=substance[[1]];
			identifiers=ToString[#["Name"]]&/@substance[[2]]
		)];
	),
	(
		Message[ServiceExecute::nparam,input[[1]],"PubChem"];
		Throw[$Failed]
	)];
	
	If[MatchQ[substance,List["name",List[__]]],
	(	
		rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawSubstances",{"namespace" -> namespace, "identifiers" -> #, "operation" -> "record", "format" -> "JSON"}],"RawJSON"]]&/@identifiers
	),
	(
		rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawSubstances",{"namespace" -> namespace, "identifiers" -> identifiers, "operation" -> "record", "format" -> "JSON"}],"RawJSON"]]
	)];
	If[MatchQ[substance,List["name",List[__]]],
	(
		If[MatchQ[#,$Failed],
   		(
    	  	Message[ServiceExecute::serrormsg,""];
    	   	Throw[$Failed]
 		)]&/@rawdata;
 		If[KeyExistsQ[#,"Fault"],
   		(
    	  	Message[ServiceExecute::serrormsg,"Message" /. ("Fault" /. #)];
    	   	Throw[$Failed]
 		)]&/@rawdata;
 	),
 	(
		If[MatchQ[rawdata,$Failed],
   		(
    	  	Message[ServiceExecute::serrormsg,""];
    	   	Throw[$Failed]
 		)];
 		If[KeyExistsQ[rawdata,"Fault"],
   		(
    	  	Message[ServiceExecute::serrormsg,"Message" /. ("Fault" /. rawdata)];
    	   	Throw[$Failed]
 		)];
 	)];
	rawdata = FixedPoint[Normal, rawdata];
 	rawdata = ("PC_Substances" /. rawdata)/.Prepend[(Rule @@@ Transpose[List[ToLowerCase[allxrefs], allxrefs]]),Rule["regid","RegistryID"]]/.{Rule["id", List[Rule["id", a_]]] :> Rule["id", a], Rule["conformers", List[d_List]] :> Rule["conformers", d],Rule["xref", b : List[__]] :> Rule["xref", Flatten[b]]} /. {Rule["coords", List[d_List]] :> Rule["coords", d],Rule["id", List[Rule["id", a_],b__]] :> Rule["id", List[a,b]]};
	withCamelTitles=Replace[rawdata, Rule[a_, b_] :> Rule[camelCase[a], b], Infinity] /. {"Id" -> "ID", "Cid" -> "CompoundID","Sid" -> "SubstanceID", "Aid" -> "AssayID", "Aid1" -> "AssayID1", "Aid2" -> "AssayID2", "Props" -> "Properties", "Urn" -> "URN","SourceId"->"SourceID","Xref"->"CrossReference","Db"->"DB","Str"->"STR"};
	Dataset[Replace[Association@@@withCamelTitles, r : {__Rule} :> Association[r], -1]]
]

pubchemcookeddata["SubstanceAssaySummary", id_,args_]:=Block[{rawdata,newparams,input,substance,namespace,identifiers,prop,params={},invalidParameters,withCamelTitles,col,row},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"SubstanceID","Name"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"PubChem"]&/@invalidParameters;
		Throw[$Failed]
	)];
	input = Select[Keys[newparams],MemberQ[{"SubstanceID","Name"},#]&]; 
	If[Length[input]>1,
	(
		Message[ServiceExecute::niden];
		Throw[$Failed]
	)];
	
	Switch[input,
	{"SubstanceID"},
	(
		newparams=newparams/.{Rule["SubstanceID",a_]:>Rule["Substance",{"SubstanceID",a}]}
	),{"Name"},
	(
		newparams=newparams/.{Rule["Name",a_]:>Rule["Substance",{"Name",a}]}
	),{},
	(
		Message[ServiceExecute::nparam,"SubstanceID or Name","PubChem"];
		Throw[$Failed]
	)];
	
	If[KeyExistsQ[newparams,"Substance"],
	(
		substance="Substance"/.newparams;
		If[!MatchQ[substance,List["SubstanceID",_String|_Integer|List[__String]|List[__Integer]]|List["Name",_String|Entity["Chemical",_]]],
		(	
			Message[ServiceExecute::nval,input[[1]],"PubChem"];
			Throw[$Failed]
		)];
		substance={substance[[1]]/.{"SubstanceID"->"sid","Name"->"name"},substance[[2]]};
		
		If[MatchQ[substance,List[_String,_String|_Integer]],
		(	
			namespace=substance[[1]];
			identifiers=ToString[substance[[2]]]
		)];
		If[MatchQ[substance,List["sid",List[__String]|List[__Integer]]],
		(	
			namespace=substance[[1]];
			identifiers=StringJoin[Riffle[ToString/@substance[[2]],","]]
		)];
		If[MatchQ[substance,List["name",Entity["Chemical",_]]],
		(
			namespace=substance[[1]];
			identifiers=ToString[substance[[2]]["Name"]]
		)]
	),
	(
		Message[ServiceExecute::nparam,input[[1]],"PubChem"];
		Throw[$Failed]
	)];
	rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawSubstances",{"namespace" -> namespace, "identifiers" -> identifiers, "operation" -> "assaysummary", "format" -> "JSON"}],"RawJSON"]];
	If[MatchQ[rawdata,$Failed],
   	(
      	Message[ServiceExecute::serrormsg,""];
       	Throw[$Failed]
 	)];
 	If[KeyExistsQ[rawdata,"Fault"],
   	(
      	Message[ServiceExecute::serrormsg,"Message" /. ("Fault" /. rawdata)];
       	Throw[$Failed]
 	)];
 	rawdata = FixedPoint[Normal, rawdata];
 	col = "Column" /. ("Columns" /. ("Table" /. rawdata));
 	row = ("Cell" /. ("Row" /. ("Table" /. rawdata))) /. {"" -> Missing["NotAvailable"]};
 	Dataset[Association @@@ Reverse[Inner[Rule, row, col, List], 3]]
]

pubchemcookeddata["SubstanceCrossReferences", id_,args_]:=Block[{rawdata,newparams,input,substance,namespace,identifiers,xref,xref2,params={},invalidParameters,withCamelTitles},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"SubstanceID","Name","CrossReference"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"PubChem"]&/@invalidParameters;
		Throw[$Failed]
	)];
	input = Select[Keys[newparams],MemberQ[{"SubstanceID","Name"},#]&]; 
	If[Length[input]>1,
	(
		Message[ServiceExecute::niden];
		Throw[$Failed]
	)];
	
	Switch[input,
	{"SubstanceID"},
	(
		newparams=newparams/.{Rule["SubstanceID",a_]:>Rule["Substance",{"SubstanceID",a}]}
	),{"Name"},
	(
		newparams=newparams/.{Rule["Name",a_]:>Rule["Substance",{"Name",a}]}
	),{},
	(
		Message[ServiceExecute::nparam,"SubstanceID or Name","PubChem"];
		Throw[$Failed]
	)];
	
	If[KeyExistsQ[newparams,"Substance"],
	(
		substance="Substance"/.newparams;		
		If[!MatchQ[substance,List[_String,_String|_Integer|_Entity|List[__Entity]|List[__String]|List[__Integer]]],
		(	
			Message[ServiceExecute::nval,input[[1]],"PubChem"];
			Throw[$Failed]
		)];
		If[!((StringMatchQ[substance[[1]], "SubstanceID"|"Name"]&&MatchQ[substance,List[_String,_String|_Integer|List[__String]|List[__Integer]]])||(MatchQ[substance,List["Name",Entity["Chemical",_]]])||(MatchQ[substance,List["Name",List[Entity["Chemical", _] ..]]])),
		(	
			Message[ServiceExecute::nval,input[[1]],"PubChem"];
			Throw[$Failed]
		)];
		substance={substance[[1]]/.{"SubstanceID"->"sid","Name"->"name"},substance[[2]]};
		If[MatchQ[substance,List[_String,_String|_Integer]],
		(	
			namespace=substance[[1]];
			identifiers=ToString[substance[[2]]]
		)];
		If[MatchQ[substance,List["sid",List[__String]|List[__Integer]]],
		(	
			namespace=substance[[1]];
			identifiers=StringJoin[Riffle[ToString/@substance[[2]],","]]
		)];
		If[MatchQ[substance,List["name",List[__String]|List[__Integer]]],
		(	
			namespace=substance[[1]];
			identifiers=ToString/@substance[[2]]
		)];
		If[MatchQ[substance,List["name",Entity["Chemical",_]]],
		(	
			namespace=substance[[1]];
			identifiers=ToString[substance[[2]]["Name"]]
		)];
		If[MatchQ[substance,List["name",List[Entity["Chemical",_]..]]],
		(	
			namespace=substance[[1]];
			identifiers=ToString[#["Name"]]&/@substance[[2]]
		)];
	),
	(
		Message[ServiceExecute::nparam,input[[1]],"PubChem"];
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"CrossReference"],
	(
		If[!(StringQ["CrossReference"/.newparams]||MatchQ["CrossReference"/.newparams,List[__String] | All]),
		(	
			Message[ServiceExecute::nval,"CrossReference","PubChem"];
			Throw[$Failed]
		)];
		If[!MatchQ["CrossReference"/.newparams,All],
		(
			xref=Flatten[{"CrossReference"/.newparams}];
			If[!(And@@StringMatchQ[xref, Alternatives@@allxrefs]),
			(	
				Message[ServiceExecute::nval,"CrossReference","PubChem"];
				Throw[$Failed]
			)];		
			xref2=StringJoin[Prepend[Riffle[xref,","],"xrefs/"]]
		),
		(
			xref=allxrefs;
			xref2=StringJoin[Prepend[Riffle[xref,","],"xrefs/"]]
		)]
	),
	(
		xref=allxrefs;
		xref2=StringJoin[Prepend[Riffle[xref,","],"xrefs/"]]
	)];
	
	If[MatchQ[substance,List["name",List[__]]],
	(	
		rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawSubstances",{"namespace" -> namespace, "identifiers" -> #, "operation" -> xref2, "format" -> "JSON"}],"RawJSON"]]&/@identifiers
	),
	(
		rawdata = Quiet[ImportString[ServiceExecute["PubChem","RawSubstances",{"namespace" -> namespace, "identifiers" -> identifiers, "operation" -> xref2, "format" -> "JSON"}],"RawJSON"]]
	)];
	If[MatchQ[substance,List["name",List[__]]],
	(
		If[MatchQ[#,$Failed],
   		(
    	  	Message[ServiceExecute::serrormsg,""];
    	   	Throw[$Failed]
 		)]&/@rawdata;
 		If[KeyExistsQ[#,"Fault"],
   		(
    	  	Message[ServiceExecute::serrormsg,"Message" /. ("Fault" /. #)];
    	   	Throw[$Failed]
 		)]&/@rawdata;
 	),
 	(
		If[MatchQ[rawdata,$Failed],
   		(
    	  	Message[ServiceExecute::serrormsg,""];
    	   	Throw[$Failed]
 		)];
 		If[KeyExistsQ[rawdata,"Fault"],
   		(
    	  	Message[ServiceExecute::serrormsg,"Message" /. ("Fault" /. rawdata)];
    	   	Throw[$Failed]
 		)];
 	)];
 	rawdata = FixedPoint[Normal, rawdata];
 	rawdata = "Information" /. ("InformationList" /. rawdata);
	withCamelTitles=Association @@@(rawdata/.{"SID"->"SubstanceID"});
	withCamelTitles=Join[#, Association[Replace[DeleteCases[Sort[Append[xref, "SubstanceID"]], Alternatives @@ Sort[Keys[#]]], {a_ :> Rule[a, Missing["NotAvailable"]]}, 1]]] & /@ withCamelTitles;
	Dataset[withCamelTitles][All,Flatten[{"SubstanceID",xref}]]
]

pubchemcookeddata["SubstanceSDF", id_,args_]:=Block[{rawdata,newparams,input,substance,namespace,type,identifiers,params={},invalidParameters,mol,elem},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"SubstanceID","Name"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"PubChem"]&/@invalidParameters;
		Throw[$Failed]
	)];	
	input = Select[Keys[newparams],MemberQ[{"SubstanceID","Name"},#]&]; 
	If[Length[input]>1,
	(
		Message[ServiceExecute::niden];
		Throw[$Failed]
	)];
	
	Switch[input,
	{"SubstanceID"},
	(
		newparams=newparams/.{Rule["SubstanceID",a_]:>Rule["Substance",{"SubstanceID",a}]}
	),{"Name"},
	(
		newparams=newparams/.{Rule["Name",a_]:>Rule["Substance",{"Name",a}]}
	),{},
	(
		Message[ServiceExecute::nparam,"SubstanceID or Name","PubChem"];
		Throw[$Failed]
	)];
	
	If[KeyExistsQ[newparams,"Substance"],
	(
		substance="Substance"/.newparams;
		If[!MatchQ[substance,List["SubstanceID",_String|_Integer]|List["Name",_String|Entity["Chemical",_]]],
		(	
			Message[ServiceExecute::nval,input[[1]],"PubChem"];
			Throw[$Failed]
		)];
		namespace=substance[[1]]/.{"SubstanceID"->"sid","Name"->"name"};
		If[!MatchQ[substance,List["name",Entity["Chemical",_]]],
		(
			identifiers=ToString[substance[[2]]]
		),
		(
			identifiers=ToString[substance[[2]]["Name"]]
		)]
	),
	(
		Message[ServiceExecute::nparam,input[[1]],"PubChem"];
		Throw[$Failed]
	)];
	rawdata = ServiceExecute["PubChem","RawSubstanceSDF",{"namespace" -> namespace, "identifiers" -> identifiers}];
	elem = ImportString[rawdata, {"SDF", "Elements"}];
	(*mol = StringSplit[StringReplace[rawdata, "$$$$\n" -> "$$$$SPLIT"], "SPLIT"];*)
	rawdata = Append[Rule[#, ImportString[rawdata, {"SDF", #}]] & /@ elem, Rule["MOL", {rawdata}]];

	If[MatchQ[#,Rule[_,$Failed]],
   	(
      	Message[ServiceExecute::serrormsg,""];
       	Throw[$Failed]
 	)]&/@rawdata;
 	Dataset[Association @@ rawdata]	
]

pubchemcookeddata[___]:=$Failed

pubchemrawdata[___]:=$Failed

pubchemsendmessage[args_]:=$Failed

End[]

End[]

SetAttributes[{},{ReadProtected, Protected}];


(* Return two functions to define oauthservicedata, oauthcookeddata  *)

{PubChemAPI`Private`pubchemdata,PubChemAPI`Private`pubchemcookeddata,PubChemAPI`Private`pubchemsendmessage,PubChemAPI`Private`pubchemrawdata}
