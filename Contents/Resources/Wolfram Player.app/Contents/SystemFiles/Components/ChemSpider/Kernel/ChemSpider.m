
Begin["ChemSpiderAPI`"]

ServiceExecute::niden = "Only one identifier is allowed";
ServiceExecute::nsparam = "Only one search parameter is allowed";
ServiceExecute::siden = "The same entry identifier was requested";

Begin["`Private`"]

(******************************* ChemSpider *************************************)

(* Authentication information *)

chemspiderdata[]:={
		"ServiceName" 		-> "ChemSpider",
        "URLFetchFun"		:> (With[{params=Lookup[{##2},"Parameters",{}]},
        							URLFetch[#1,"ContentData", Sequence@@FilterRules[{##2},Except["Parameters"]], 
									"Parameters" -> (params/."apikey"->"token")]]&),
        "ClientInfo"		:> OAuthDialogDump`Private`MultipleKeyDialog["ChemSpider",{"Security Token"->"token"},
                                        "https://www.chemspider.com/UserProfile.aspx","http://www.rsc.org/help-legal/legal/terms-conditions/"],
	 	"Gets"				-> {"Search","CompoundInformation","CompoundThumbnail","Databases","ExtendedCompoundInformation","AllSpectraInformation","CompoundSpectraInformation","SpectrumInformation","InChIKeyQ","GetIdentifier"(*,"MOLToInChI","MOLToInChIKey","IDToMOL","RecordToMOL","MOLToID"*)},
	 	"RawGets"			-> {"RawGetDatabases", "RawGetExtendedCompoundInfo", "RawGetRecordMOL", "RawSearchByFormula2","RawSearchByMass2",
	 		"RawAsyncSimpleSearch","RawAsyncSimpleSearchOrdered","RawGetAsyncSearchResults","RawGetCompoundInfo","RawGetCompoundThumbnail","RawMol2CSID","RawSimpleSearch","RawGetAllSpectraInfo",
	 		"RawGetCompoundSpectraInfo","RawGetSpectrumInfo","RawCSIDToMol","RawIsValidInChIKey","RawMolToInChI","RawMolToInChIKey","RawInChIToMol","RawInChIToInChIKey","RawInChIToCSID","RawInChIKeyToMol","RawInChIKeyToInChI","RawInChIKeyToCSID"},
	 	"Posts"				-> {},
	 	"RawPosts"			-> {},
 		"Information"		-> "Import ChemSpider API data to the Wolfram Language"
 		}

chemspiderimport[rawdata_]:=FromCharacterCode[rawdata, "UTF-8"]

(* Raw *)
chemspiderdata["RawGetDatabases"] := {
        "URL"				-> "http://www.chemspider.com/MassSpecAPI.asmx/GetDatabases",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> chemspiderimport
        }

chemspiderdata["RawGetExtendedCompoundInfo"] := {
		"URL"				-> "http://www.chemspider.com/MassSpecAPI.asmx/GetExtendedCompoundInfo",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"csid"},
        "RequiredParameters"-> {"csid"},
        "ResultsFunction"	-> chemspiderimport
        }

chemspiderdata["RawGetRecordMol"] := {
		"URL"				-> "http://www.chemspider.com/MassSpecAPI.asmx/GetRecordMol",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"csid","calc3d"},
        "RequiredParameters"-> {"csid","calc3d"},
        "ResultsFunction"	-> chemspiderimport
        }
        
chemspiderdata["RawSearchByFormula2"] := {
		"URL"				-> "http://www.chemspider.com/MassSpecAPI.asmx/SearchByFormula2",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"formula"},
        "RequiredParameters"-> {"formula"},
        "ResultsFunction"	-> chemspiderimport
        }        

chemspiderdata["RawSearchByMass2"] := {
		"URL"				-> "http://www.chemspider.com/MassSpecAPI.asmx/SearchByMass2",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"mass","range"},
        "RequiredParameters"-> {"mass","range"},
        "ResultsFunction"	-> chemspiderimport
        }        

chemspiderdata["RawAsyncSimpleSearch"] := {
		"URL"				-> "http://www.chemspider.com/Search.asmx/AsyncSimpleSearch",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"query"},
        "RequiredParameters"-> {"query"},
        "ResultsFunction"	-> chemspiderimport
        }        

chemspiderdata["RawAsyncSimpleSearchOrdered"] := {
		"URL"				-> "http://www.chemspider.com/Search.asmx/AsyncSimpleSearchOrdered",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"query","orderby","orderdirection"},
        "RequiredParameters"-> {"query","orderby","orderdirection"},
        "ResultsFunction"	-> chemspiderimport
        }        

chemspiderdata["RawGetAsyncSearchResults"] := {
		"URL"				-> "http://www.chemspider.com/Search.asmx/GetAsyncSearchResultPart",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"rid","start","count"},
        "RequiredParameters"-> {"rid","start","count"},
        "ResultsFunction"	-> chemspiderimport
        }

chemspiderdata["RawGetCompoundInfo"] := {
		"URL"				-> "http://www.chemspider.com/Search.asmx/GetCompoundInfo",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"csid"},
        "RequiredParameters"-> {"csid"},
        "ResultsFunction"	-> chemspiderimport
        }        

chemspiderdata["RawGetCompoundThumbnail"] := {
		"URL"				-> "http://www.chemspider.com/Search.asmx/GetCompoundThumbnail",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"id"},
        "RequiredParameters"-> {"id"},
        "ResultsFunction"	-> chemspiderimport
        }        

chemspiderdata["RawMol2CSID"] := {
		"URL"				-> "http://www.chemspider.com/Search.asmx/Mol2CSID",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"options","mol"},
        "RequiredParameters"-> {"options","mol"},
        "ResultsFunction"	-> chemspiderimport
        }        

chemspiderdata["RawSimpleSearch"] := {
		"URL"				-> "http://www.chemspider.com/Search.asmx/SimpleSearch",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"query"},
        "RequiredParameters"-> {"query"},
        "ResultsFunction"	-> chemspiderimport
        }        

chemspiderdata["RawGetAllSpectraInfo"] := {
		"URL"				-> "http://www.chemspider.com/Spectra.asmx/GetAllSpectraInfo",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> chemspiderimport
        }        

chemspiderdata["RawGetCompoundSpectraInfo"] := {
		"URL"				-> "http://www.chemspider.com/Spectra.asmx/GetCompoundSpectraInfo",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"csid"},
        "RequiredParameters"-> {"csid"},
        "ResultsFunction"	-> chemspiderimport
        }        

chemspiderdata["RawGetSpectrumInfo"] := {
		"URL"				-> "http://www.chemspider.com/Spectra.asmx/GetSpectrumInfo",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"spc_id"},
        "RequiredParameters"-> {"spc_id"},
        "ResultsFunction"	-> chemspiderimport
        }        

chemspiderdata["RawCSIDToMol"] := {
		"URL"				-> "http://www.chemspider.com/InChI.asmx/CSIDToMol",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"csid"},
        "RequiredParameters"-> {"csid"},
        "ResultsFunction"	-> chemspiderimport
        }        

chemspiderdata["RawIsValidInChIKey"] := {
		"URL"				-> "http://www.chemspider.com/InChI.asmx/IsValidInChIKey",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"inchi_key"},
        "RequiredParameters"-> {"inchi_key"},
        "ResultsFunction"	-> chemspiderimport
        }        

chemspiderdata["RawMolToInChI"] := {
		"URL"				-> "http://www.chemspider.com/InChI.asmx/MolToInChI",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"mol"},
        "RequiredParameters"-> {"mol"},
        "ResultsFunction"	-> chemspiderimport
        }        

chemspiderdata["RawMolToInChIKey"] := {
		"URL"				-> "http://www.chemspider.com/InChI.asmx/MolToInChIKey",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"mol"},
        "RequiredParameters"-> {"mol"},
        "ResultsFunction"	-> chemspiderimport
        }        

chemspiderdata["RawInChIToMol"] := {
		"URL"				-> "http://www.chemspider.com/InChI.asmx/InChIToMol",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"inchi"},
        "RequiredParameters"-> {"inchi"},
        "ResultsFunction"	-> chemspiderimport
        }

chemspiderdata["RawInChIToInChIKey"] := {
		"URL"				-> "http://www.chemspider.com/InChI.asmx/InChIToInChIKey",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"inchi"},
        "RequiredParameters"-> {"inchi"},
        "ResultsFunction"	-> chemspiderimport
        }

chemspiderdata["RawInChIToCSID"] := {
		"URL"				-> "http://www.chemspider.com/InChI.asmx/InChIToCSID",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"inchi"},
        "RequiredParameters"-> {"inchi"},
        "ResultsFunction"	-> chemspiderimport
        }
        
chemspiderdata["RawInChIKeyToMol"] := {
		"URL"				-> "http://www.chemspider.com/InChI.asmx/InChIKeyToMol",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"inchi_key"},
        "RequiredParameters"-> {"inchi_key"},
        "ResultsFunction"	-> chemspiderimport
        }

chemspiderdata["RawInChIKeyToInChI"] := {
		"URL"				-> "http://www.chemspider.com/InChI.asmx/InChIKeyToInChI",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"inchi_key"},
        "RequiredParameters"-> {"inchi_key"},
        "ResultsFunction"	-> chemspiderimport
        }

chemspiderdata["RawInChIKeyToCSID"] := {
		"URL"				-> "http://www.chemspider.com/InChI.asmx/InChIKeyToCSID",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"inchi_key"},
        "RequiredParameters"-> {"inchi_key"},
        "ResultsFunction"	-> chemspiderimport
        }
(* Cooked *)

camelCase[text_] := Module[{split, partial}, (
    split = StringSplit[text, {" ","_","-"}];
    partial = Prepend[Rest[Characters[#]], ToUpperCase[Characters[#][[1]]]] & /@ split;
    StringJoin[partial]
    )]


searchOld[args_]:=Block[{rawdata,newparams,params={"orderdirection" -> "eDescending"},invalidParameters,withCamelTitles,maxitems,startindex,length,transactionid},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"Query","SortBy","MaxItems","StartIndex"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"ChemSpider"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"Query"],
	(
		If[!StringQ["Query"/.newparams],
		(	
			Message[ServiceExecute::nval,"Query","ChemSpider"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["query","Query"/.newparams]]
	),
	(
		Message[ServiceExecute::nparam,"Query","ChemSpider"];
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"MaxItems"],
	(
		If[!((IntegerQ["MaxItems"/.newparams]&&("MaxItems"/.newparams)>0)||MatchQ["MaxItems"/.newparams,All]),
		(	
			Message[ServiceExecute::nval,"MaxItems","ChemSpider"];
			Throw[$Failed]
		)];
		maxitems="MaxItems"/.newparams
	),
  	(
  		maxitems=10
  	)];
	If[KeyExistsQ[newparams,"StartIndex"],
	(
		If[!(IntegerQ["StartIndex"/.newparams]&&("StartIndex"/.newparams)>0),
		(	
			Message[ServiceExecute::nval,"StartIndex","ChemSpider"];
			Throw[$Failed]
		)];
		startindex="StartIndex"/.newparams
	),
  	(
  		startindex=1
  	)];
  	If[KeyExistsQ[newparams,"SortBy"],
	(
		If[!StringMatchQ[ToString["SortBy" /. newparams], "ID"|"MolecularWeight"|"ReferenceCount"|"DataSourceCount"|"PubMedCount"|"RSCCount"],
		(	
			Message[ServiceExecute::nval,"SortBy","ChemSpider"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["orderby",(ToString["SortBy" /. newparams]/.{"ID"->"eCSID","MolecularWeight"->"eMolecularWeight","ReferenceCount"->"eReferenceCount","DataSourceCount"->"eDataSourceCount","PubMedCount"->"ePubMedCount","RSCCount"->"eRscCount"})]]
	),
  	(
  		params = Append[params,Rule["orderby","eDataSourceCount"]]
  	)];
	transactionid = Quiet[ImportString[ServiceExecute["ChemSpider","RawAsyncSimpleSearchOrdered",params],"XML"]];
	If[MatchQ[transactionid,$Failed],
   	(
      	Message[ServiceExecute::serrormsg,""];
       	Throw[$Failed]
 	)];
 	transactionid = ("string" /. Replace[Cases[transactionid, XMLElement[_, _, _]], {XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity])[[1]];
 	rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawGetAsyncSearchResults",{"rid"->transactionid,"start" -> "0", "count" -> "-1"}],"XML"]];
	If[MatchQ[rawdata,$Failed],
   	(
      	Message[ServiceExecute::serrormsg,""];
       	Throw[$Failed]
 	)];
 	rawdata="ArrayOfInt" /.Replace[Cases[rawdata,  XMLElement[_, _, _]], {XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity];
 	
	withCamelTitles=Replace[rawdata,{Rule["int", {b_}] :> Rule["ID", b]}, Infinity];
	If[MatchQ[maxitems,All],
	(
		If[MatchQ[Length[withCamelTitles],1],
		(
			Dataset[Association[withCamelTitles[[1]]]]
		),
		(
			Dataset[Association /@ withCamelTitles]
		)]
	),
	(
		withCamelTitles=Partition[withCamelTitles,UpTo[maxitems]];
		length=Length[withCamelTitles];
		If[startindex>length,
		(
			Dataset[{}]
		),
		(
			If[MatchQ[length,1],
			(
				Dataset[Association[withCamelTitles[[startindex]][[1]]]]
			),
			(
				Dataset[Association /@ withCamelTitles[[startindex]]]
			)]
		)]
	)]
]

searchByFormula[args_]:=Block[{rawdata,newparams,params={},invalidParameters,withCamelTitles,maxitems,startindex,lenght},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"Formula","MaxItems","StartIndex"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"ChemSpider"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"Formula"],
	(
		If[!StringQ["Formula"/.newparams],
		(	
			Message[ServiceExecute::nval,"Formula","ChemSpider"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["formula","Formula"/.newparams]]
	),
	(
		Message[ServiceExecute::nparam,"Formula","ChemSpider"];
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"MaxItems"],
	(
		If[!((IntegerQ["MaxItems"/.newparams]&&("MaxItems"/.newparams)>0)||MatchQ["MaxItems"/.newparams,All]),
		(	
			Message[ServiceExecute::nval,"MaxItems","ChemSpider"];
			Throw[$Failed]
		)];
		maxitems="MaxItems"/.newparams
	),
  	(
  		maxitems=10
  	)];
	If[KeyExistsQ[newparams,"StartIndex"],
	(
		If[!(IntegerQ["StartIndex"/.newparams]&&("StartIndex"/.newparams)>0),
		(	
			Message[ServiceExecute::nval,"StartIndex","ChemSpider"];
			Throw[$Failed]
		)];
		startindex="StartIndex"/.newparams
	),
  	(
  		startindex=1
  	)];
	rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawSearchByFormula2",params],"XML"]];
	If[MatchQ[rawdata,$Failed],
   	(
      	Message[ServiceExecute::serrormsg,""];
       	Throw[$Failed]
 	)];
	withCamelTitles=("ArrayOfString" /. Replace[Cases[rawdata, XMLElement[_, _, _]], {XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity]) /. {Rule[c_, {d_}] :> Rule["ID", d]};
	If[MatchQ[maxitems,All],
	(
		If[MatchQ[Length[withCamelTitles],1],
		(
			Dataset[Association[withCamelTitles[[1]]]]
		),
		(
			Dataset[Association /@ withCamelTitles]
		)]
	),
	(
		withCamelTitles=Partition[withCamelTitles,UpTo[maxitems]];
		length=Length[withCamelTitles];
		If[startindex>length,
		(
			Dataset[{}]
		),
		(
			If[MatchQ[length,1],
			(
				Dataset[Association[withCamelTitles[[startindex]][[1]]]]
			),
			(
				Dataset[Association /@ withCamelTitles[[startindex]]]
			)]
		)]
	)]
]

searchByMass[args_]:=Block[{rawdata,newparams,params={},invalidParameters,withCamelTitles,maxitems,startindex,length},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"Mass","Range","MaxItems","StartIndex"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"ChemSpider"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"Mass"],
	(
		If[!NumberQ["Mass"/.newparams],
		(	
			Message[ServiceExecute::nval,"Mass","ChemSpider"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["mass",ToString["Mass"/.newparams]]]
	),
	(
		Message[ServiceExecute::nparam,"Mass","ChemSpider"];
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"Range"],
	(
		If[!NumberQ["Range"/.newparams],
		(	
			Message[ServiceExecute::nval,"Range","ChemSpider"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["range",ToString["Range"/.newparams]]]
	),
	(
		Message[ServiceExecute::nparam,"Range","ChemSpider"];
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"MaxItems"],
	(
		If[!((IntegerQ["MaxItems"/.newparams]&&("MaxItems"/.newparams)>0)||MatchQ["MaxItems"/.newparams,All]),
		(	
			Message[ServiceExecute::nval,"MaxItems","ChemSpider"];
			Throw[$Failed]
		)];
		maxitems="MaxItems"/.newparams
	),
  	(
  		maxitems=10
  	)];
	If[KeyExistsQ[newparams,"StartIndex"],
	(
		If[!(IntegerQ["StartIndex"/.newparams]&&("StartIndex"/.newparams)>0),
		(	
			Message[ServiceExecute::nval,"StartIndex","ChemSpider"];
			Throw[$Failed]
		)];
		startindex="StartIndex"/.newparams
	),
  	(
  		startindex=1
  	)];
	rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawSearchByMass2",params],"XML"]];
	If[MatchQ[rawdata,$Failed],
   	(
      	Message[ServiceExecute::serrormsg,""];
       	Throw[$Failed]
 	)];
	withCamelTitles=("ArrayOfString" /. Replace[Cases[rawdata, XMLElement[_, _, _]], {XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity]) /. {Rule[c_, {d_}] :> Rule["ID", d]};
	If[MatchQ[maxitems,All],
	(
		If[MatchQ[Length[withCamelTitles],1],
		(
			Dataset[Association[withCamelTitles[[1]]]]
		),
		(
			Dataset[Association /@ withCamelTitles]
		)]
	),
	(
		withCamelTitles=Partition[withCamelTitles,UpTo[maxitems]];
		length=Length[withCamelTitles];
		If[startindex>length,
		(
			Dataset[{}]
		),
		(
			If[MatchQ[length,1],
			(
				Dataset[Association[withCamelTitles[[startindex]][[1]]]]
			),
			(
				Dataset[Association /@ withCamelTitles[[startindex]]]
			)]
		)]
	)]
]

chemspidercookeddata["Search", id_,args_]:=Block[{rawdata,newparams,params={},invalidParameters,withCamelTitles,input,output},
	newparams=args/.{MaxItems:>"MaxItems"};
	invalidParameters = Select[Keys[newparams],!MemberQ[{"Query","MaxItems","StartIndex","SortBy","Formula","Mass","Range"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"ChemSpider"]&/@invalidParameters;
		Throw[$Failed]
	)];
	input = Select[Keys[newparams],MemberQ[{"Query","Formula","Mass"},#]&]; 
	If[Length[input]>1,
	(
		Message[ServiceExecute::nsparam];
		Throw[$Failed]
	)];
	
	Switch[input,
	{"Query"},
	(
		searchOld[newparams]
	),{"Formula"},
	(
		searchByFormula[newparams]
	),{"Mass"},
	(
		searchByMass[newparams]
	),{},
	(
		Message[ServiceExecute::nparam,"Query, Formula or Mass","ChemSpider"];
		Throw[$Failed]
	)]
]



chemspidercookeddata["CompoundInformation", id_,args_]:=Block[{rawdata,newparams,params={},invalidParameters,withCamelTitles},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"ID"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"ChemSpider"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"ID"],
	(
		If[!(StringQ["ID"/.newparams]||IntegerQ["ID"/.newparams]),
		(	
			Message[ServiceExecute::nval,"ID","ChemSpider"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["csid",ToString["ID"/.newparams]]]
	),
	(
		Message[ServiceExecute::nparam,"ID","ChemSpider"];
		Throw[$Failed]
	)];
	rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawGetCompoundInfo",params],"XML"]];
	If[MatchQ[rawdata,$Failed],
   	(
      	Message[ServiceExecute::serrormsg,""];
       	Throw[$Failed]
 	)];
	withCamelTitles=("CompoundInfo" /. Replace[Cases[rawdata, XMLElement[_, _, _]], {XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity]) /. {Rule[c_, {d_}] :> Rule[c, d], Rule["CSID", e_] :> Rule["ID", e]};
	Dataset[Association[withCamelTitles]]
]

chemspidercookeddata["CompoundThumbnail", id_,args_]:=Block[{rawdata,newparams,params={},invalidParameters},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"ID"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"ChemSpider"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"ID"],
	(
		If[!(StringQ["ID"/.newparams]||IntegerQ["ID"/.newparams]),
		(	
			Message[ServiceExecute::nval,"ID","ChemSpider"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["id",ToString["ID"/.newparams]]]
	),
	(
		Message[ServiceExecute::nparam,"ID","ChemSpider"];
		Throw[$Failed]
	)];
	rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawGetCompoundThumbnail",params],"XML"]];
	If[MatchQ[rawdata,$Failed],
   	(
      	Message[ServiceExecute::serrormsg,""];
       	Throw[$Failed]
 	)];
	rawdata =  ("base64Binary" /. Replace[Cases[rawdata, XMLElement[_, _, _]], {XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity])[[1]];
	ImportString[rawdata, "Base64"]
]

chemspidercookeddata["Databases", id_,args_]:=Block[{rawdata,newparams,params={},invalidParameters},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"ChemSpider"]&/@invalidParameters;
		Throw[$Failed]
	)];
	rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawGetDatabases",params],"XML"]];
	If[MatchQ[rawdata,$Failed],
   	(
      	Message[ServiceExecute::serrormsg,""];
       	Throw[$Failed]
 	)];
 	("ArrayOfString" /. Replace[Cases[rawdata, XMLElement[_, _, _]], {XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity])[[All, 2, 1]]
]

chemspidercookeddata["ExtendedCompoundInformation", id_,args_]:=Block[{rawdata,newparams,params={},invalidParameters,withCamelTitles},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"ID"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"ChemSpider"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"ID"],
	(
		If[!(StringQ["ID"/.newparams]||IntegerQ["ID"/.newparams]),
		(	
			Message[ServiceExecute::nval,"ID","ChemSpider"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["csid",ToString["ID"/.newparams]]]
	),
	(
		Message[ServiceExecute::nparam,"ID","ChemSpider"];
		Throw[$Failed]
	)];
	rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawGetExtendedCompoundInfo",params],"XML"]];
	If[MatchQ[rawdata,$Failed],
   	(
      	Message[ServiceExecute::serrormsg,""];
       	Throw[$Failed]
 	)];
	withCamelTitles=("ExtendedCompoundInfo" /. Replace[Cases[rawdata, XMLElement[_, _, _]], {XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity]) /. {Rule[c_, {d_}] :> Rule[c, d]};
	Dataset[Association[withCamelTitles]]
]


chemspidercookeddata["AllSpectraInformation", id_,args_]:=Block[{rawdata,newparams,params={},invalidParameters,withCamelTitles},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"ChemSpider"]&/@invalidParameters;
		Throw[$Failed]
	)];
	rawdata=ServiceExecute["ChemSpider","RawGetAllSpectraInfo",params];
	If[StringMatchQ[rawdata, ___ ~~ "Unauthorized web service usage" ~~ ___],
   	(
      	Message[ServiceExecute::serrormsg,"Service is not subscribed by the user"];
       	Throw[$Failed]
 	)];
	rawdata = Quiet[ImportString[rawdata,"XML"]];
	If[MatchQ[rawdata,$Failed],
   	(
      	Message[ServiceExecute::serrormsg,""];
       	Throw[$Failed]
 	)];
 	rawdata=("ArrayOfCSSpectrumInfo" /. Replace[Cases[rawdata, XMLElement[_, _, _]], {XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity])[[All, 2]] /. {Rule[c_, {d_}] :> Rule[c, d]};
	withCamelTitles=Replace[rawdata,{Rule[a_, b_] :> Rule[camelCase[a], b]},Infinity]/.{"SpcId"->"SpectrumID","Csid"->"ID","SpcType"->"SpectrumType","OriginalUrl"->"OriginalURL",
		("SubmittedDate" -> x_) :> ("SubmittedDate" -> If[MatchQ[x, _String], DateObject[StringReplace[x, "T" -> " "], TimeZone -> 0], x])};
	Dataset[Association/@withCamelTitles]
]

chemspidercookeddata["CompoundSpectraInformation", id_,args_]:=Block[{rawdata,newparams,params={},invalidParameters,withCamelTitles},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"ID"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"ChemSpider"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"ID"],
	(
		If[!(StringQ["ID"/.newparams]||IntegerQ["ID"/.newparams]),
		(	
			Message[ServiceExecute::nval,"ID","ChemSpider"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["csid",ToString["ID"/.newparams]]]
	),
	(
		Message[ServiceExecute::nparam,"ID","ChemSpider"];
		Throw[$Failed]
	)];
	rawdata=ServiceExecute["ChemSpider","RawGetCompoundSpectraInfo",params];
	If[StringMatchQ[rawdata, ___ ~~ "Unauthorized web service usage" ~~ ___],
   	(
      	Message[ServiceExecute::serrormsg,"Service is not subscribed by the user"];
       	Throw[$Failed]
 	)];
	rawdata = Quiet[ImportString[rawdata,"XML"]];
	If[MatchQ[rawdata,$Failed],
   	(
      	Message[ServiceExecute::serrormsg,""];
       	Throw[$Failed]
 	)];
 	rawdata=("ArrayOfCSSpectrumInfo" /. Replace[Cases[rawdata, XMLElement[_, _, _]], {XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity])[[All, 2]] /. {Rule[c_, {d_}] :> Rule[c, d]};
	withCamelTitles=Replace[rawdata,{Rule[a_, b_] :> Rule[camelCase[a], b]},Infinity]/.{"SpcId"->"SpectrumID","Csid"->"ID","SpcType"->"SpectrumType","OriginalUrl"->"OriginalURL",
		("SubmittedDate" -> x_) :> ("SubmittedDate" -> If[MatchQ[x, _String], DateObject[StringReplace[x, "T" -> " "], TimeZone -> 0], x])};
	Dataset[Association/@withCamelTitles]
]

chemspidercookeddata["SpectrumInformation", id_,args_]:=Block[{rawdata,newparams,params={},invalidParameters,withCamelTitles},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"SpectrumID"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"ChemSpider"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"SpectrumID"],
	(
		If[!(StringQ["SpectrumID"/.newparams]||IntegerQ["SpectrumID"/.newparams]),
		(	
			Message[ServiceExecute::nval,"SpectrumID","ChemSpider"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["spc_id",ToString["SpectrumID"/.newparams]]]
	),
	(
		Message[ServiceExecute::nparam,"SpectrumID","ChemSpider"];
		Throw[$Failed]
	)];
	rawdata=ServiceExecute["ChemSpider","RawGetSpectrumInfo",params];
	If[StringMatchQ[rawdata, ___ ~~ "Unauthorized web service usage" ~~ ___],
   	(
      	Message[ServiceExecute::serrormsg,"Service is not subscribed by the user"];
       	Throw[$Failed]
 	)];
	rawdata = Quiet[ImportString[rawdata,"XML"]];
	If[MatchQ[rawdata,$Failed],
   	(
      	Message[ServiceExecute::serrormsg,""];
       	Throw[$Failed]
 	)];
 	rawdata = ("CSSpectrumInfo" /. Replace[Cases[rawdata, XMLElement[_, _, _]], {XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity]) /. {Rule[c_, {d_}] :> Rule[c, d]};
	withCamelTitles=Replace[rawdata,{Rule[a_, b_] :> Rule[camelCase[a], b]},Infinity]/.{"SpcId"->"SpectrumID","Csid"->"ID","SpcType"->"SpectrumType","OriginalUrl"->"OriginalURL",
		("SubmittedDate" -> x_) :> ("SubmittedDate" -> If[MatchQ[x, _String], DateObject[StringReplace[x, "T" -> " "], TimeZone -> 0], x])};
	Dataset[Association[withCamelTitles]]
]

chemspidercookeddata["InChIKeyQ", id_,args_]:=Block[{rawdata,newparams,params={},invalidParameters,withCamelTitles},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"InChIKey"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"ChemSpider"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"InChIKey"],
	(
		If[!StringQ["InChIKey"/.newparams],
		(	
			Message[ServiceExecute::nval,"InChIKey","ChemSpider"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["inchi_key",ToString["InChIKey"/.newparams]]]
	),
	(
		Message[ServiceExecute::nparam,"InChIKey","ChemSpider"];
		Throw[$Failed]
	)];
	rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawIsValidInChIKey",params],"XML"]];
	If[MatchQ[rawdata,$Failed],
   	(
      	Message[ServiceExecute::serrormsg,""];
       	Throw[$Failed]
 	)];
 	rawdata="boolean" /. Replace[Cases[rawdata, XMLElement[_, _, _]], {XMLElement[a_String, _, {b_}] :> Rule[a, b],XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity];
	rawdata/.{"true"->True,"false"->False}
]


chemspidercookeddata["GetIdentifier", id_,args_]:=Block[{rawdata,newparams,params={},invalidParameters,withCamelTitles,input,output,opts},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"MOL","InChI","InChIKey","ID","Identifier","Options"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"ChemSpider"]&/@invalidParameters;
		Throw[$Failed]
	)];
	input = Select[Keys[newparams],MemberQ[{"MOL","InChI","InChIKey","ID"},#]&]; 
	If[Length[input]>1,
	(
		Message[ServiceExecute::niden];
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"Identifier"],
	(
		If[!StringMatchQ[ToString["Identifier"/.newparams], "MOL"|"InChI"|"InChIKey"|"ID"], (*Not supporting Lists*)
		(	
			Message[ServiceExecute::nval,"Identifier","PubChem"];
			Throw[$Failed]
		)];
		output=ToString["Identifier"/.newparams]
	),
	(
		Message[ServiceExecute::nparam,"Identifier","ChemSpider"];
		Throw[$Failed]
	)];
	
	Switch[input,
	{"MOL"},
	(
		If[!(StringQ["MOL"/.newparams]||MatchQ["MOL"/.newparams,List[__String]]),  
		(	
			Message[ServiceExecute::nval,"MOL","ChemSpider"];
			Throw[$Failed]
		)];
		Switch[output,
		"MOL",
		(
			Message[ServiceExecute::siden];
			Throw[$Failed]
		),"InChI",
		(
			If[StringQ["MOL"/.newparams],
			(
				rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawMolToInChI",{"mol"->("MOL"/.newparams)}],"XML"]];
				If[MatchQ[rawdata,$Failed],
   				(
   			 	  	Message[ServiceExecute::serrormsg,""];
  		    	 	Throw[$Failed]
 				)];
 				withCamelTitles=Replace[Cases[rawdata, XMLElement[_, _, _]], {XMLElement["string", _, {b_}] :> Rule["InChI", b], XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity];
				Dataset[Association[withCamelTitles]]
			),
			(
				rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawMolToInChI",{"mol"->#}],"XML"]]&/@("MOL"/.newparams);
				If[MatchQ[#,$Failed],
   				(
   			 	  	Message[ServiceExecute::serrormsg,""];
  		    	 	Throw[$Failed]
 				)]&/@rawdata;
 				withCamelTitles=Replace[Cases[#, XMLElement[_, _, _]], {XMLElement["string", _, {b_}] :> Rule["InChI", b], XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity]&/@rawdata;
				withCamelTitles = MapThread[List, {Rule["MOL", #] & /@ ("MOL"/.newparams), withCamelTitles}];
				Dataset[Association /@ withCamelTitles]
			)]
		),"InChIKey",
		(
			If[StringQ["MOL"/.newparams],
			(
				rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawMolToInChIKey",{"mol"->("MOL"/.newparams)}],"XML"]];
				If[MatchQ[rawdata,$Failed],
   				(
      				Message[ServiceExecute::serrormsg,""];
       				Throw[$Failed]
 				)];
 				withCamelTitles=Replace[Cases[rawdata, XMLElement[_, _, _]], {XMLElement["string", _, {b_}] :> Rule["InChIKey", b], XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity];
				Dataset[Association[withCamelTitles]]
			),
			(
				rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawMolToInChIKey",{"mol"->#}],"XML"]]&/@("MOL"/.newparams);
				If[MatchQ[#,$Failed],
   				(
   			 	  	Message[ServiceExecute::serrormsg,""];
  		    	 	Throw[$Failed]
 				)]&/@rawdata;
 				withCamelTitles=Replace[Cases[#, XMLElement[_, _, _]], {XMLElement["string", _, {b_}] :> Rule["InChIKey", b], XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity]&/@rawdata;
				withCamelTitles = MapThread[List, {Rule["MOL", #] & /@ ("MOL"/.newparams), withCamelTitles}];
				Dataset[Association /@ withCamelTitles]
			)]
		),"ID",
		(
			If[KeyExistsQ[newparams,"Options"],
			(
				If[!StringMatchQ[ToString["Options" /. newparams],"ExactMatch"|"AllTautomers"|"SameSkeletonAndH"|"SameSkeleton"|"AllIsomers"],
				(	
					Message[ServiceExecute::nval,"Options","ChemSpider"];
					Throw[$Failed]
				)];
				opts = ToString["Options" /. newparams]/.{"ExactMatch"->"eExactMatch","AllTautomers"->"eAllTautomers","SameSkeletonAndH"->"eSameSkeletonAndH","SameSkeleton"->"eSameSkeleton","AllIsomers"->"eAllIsomers"}
			),
  			(
  				opts = "eExactMatch"
  			)];
			If[StringQ["MOL"/.newparams],
			(
				rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawMol2CSID",{"mol"->("MOL"/.newparams),"options"->opts}],"XML"]];
				If[MatchQ[rawdata,$Failed],
   				(
      				Message[ServiceExecute::serrormsg,""];
       				Throw[$Failed]
 				)];
 				withCamelTitles=("ArrayOfInt" /. Replace[Cases[rawdata, XMLElement[_, _, _]], {XMLElement[a_String, _, b_] :>  Rule[a, b]}, Infinity]) /. {Rule["int", {b_}] :> Rule["ID", b]};
				Dataset[GroupBy[withCamelTitles, First -> Last]]
			),
			(
				rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawMol2CSID",{"mol"->#,"options"->opts}],"XML"]]&/@("MOL"/.newparams);
				If[MatchQ[#,$Failed],
   				(
      				Message[ServiceExecute::serrormsg,""];
       				Throw[$Failed]
 				)]&/@rawdata;
 				withCamelTitles=(("ArrayOfInt" /. Replace[Cases[#, XMLElement[_, _, _]], {XMLElement[a_String, _, b_] :>  Rule[a, b]}, Infinity]) /. {Rule["int", {b_}] :> Rule["ID", b]})& /@ rawdata;
				withCamelTitles = Flatten[Normal[GroupBy[#, First -> Last] & /@ withCamelTitles]];
				withCamelTitles = MapThread[List, {Rule["MOL", #] & /@ ("MOL"/.newparams), withCamelTitles}];
				Dataset[Association /@ withCamelTitles]
			)]
		)]
	),{"InChI"},
	(
		If[!(StringQ["InChI"/.newparams]||MatchQ["InChI"/.newparams,List[__String]]), 
		(	
			Message[ServiceExecute::nval,"InChI","ChemSpider"];
			Throw[$Failed]
		)];
		Switch[output,
		"InChI",
		(
			Message[ServiceExecute::siden];
			Throw[$Failed]
		),"MOL",
		(
			If[StringQ["InChI"/.newparams],
			(
				rawdata = ServiceExecute["ChemSpider","RawInChIToMol",{"inchi"->("InChI"/.newparams)}];
 				withCamelTitles=StringReplace[rawdata, ___ ~~ "<string xmlns=\"http://www.chemspider.com/\">" ~~ a__ ~~ "</string>" ~~ ___ :> a];
				Dataset[Association["MOL"->withCamelTitles]]
			),
			(
				rawdata = ServiceExecute["ChemSpider","RawInChIToMol",{"inchi"->#}]&/@("InChI"/.newparams);
 				withCamelTitles=StringReplace[#, ___ ~~ "<string xmlns=\"http://www.chemspider.com/\">" ~~ a__ ~~ "</string>" ~~ ___ :> a]&/@rawdata;
				withCamelTitles = List["MOL"->#]&/@withCamelTitles;
				withCamelTitles = MapThread[List, {Rule["InChI", #] & /@ ("InChI"/.newparams), withCamelTitles}];
				Dataset[Association/@withCamelTitles]
			)]
		),"InChIKey",
		(
			If[StringQ["InChI"/.newparams],
			(
				rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawInChIToInChIKey",{"inchi"->("InChI"/.newparams)}],"XML"]];
				If[MatchQ[rawdata,$Failed],
   				(
      				Message[ServiceExecute::serrormsg,""];
       				Throw[$Failed]
 				)];
 				withCamelTitles=Replace[Cases[rawdata, XMLElement[_, _, _]], {XMLElement["string", _, {b_}] :> Rule["InChIKey", b], XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity];
				Dataset[Association[withCamelTitles]]
			),
			(
				rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawInChIToInChIKey",{"inchi"->#}],"XML"]]&/@("InChI"/.newparams);
				If[MatchQ[#,$Failed],
   				(
      				Message[ServiceExecute::serrormsg,""];
       				Throw[$Failed]
 				)]&/@rawdata;
 				withCamelTitles=Replace[Cases[#, XMLElement[_, _, _]], {XMLElement["string", _, {b_}] :> Rule["InChIKey", b], XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity]&/@rawdata;
				withCamelTitles = MapThread[List, {Rule["InChI", #] & /@ ("InChI"/.newparams), withCamelTitles}];
				Dataset[Association /@ withCamelTitles]
			)]
		),"ID",
		(
			If[StringQ["InChI"/.newparams],
			(
				rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawInChIToCSID",{"inchi"->("InChI"/.newparams)}],"XML"]];
				If[MatchQ[rawdata,$Failed],
   				(
      				Message[ServiceExecute::serrormsg,""];
       				Throw[$Failed]
 				)];
 				withCamelTitles=Replace[Cases[rawdata, XMLElement[_, _, _]], {XMLElement["string", _, {b_}] :> Rule["ID", b], XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity];
				Dataset[Association[withCamelTitles]]
			),
			(
				rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawInChIToCSID",{"inchi"->#}],"XML"]]&/@("InChI"/.newparams);
				If[MatchQ[#,$Failed],
   				(
      				Message[ServiceExecute::serrormsg,""];
       				Throw[$Failed]
 				)]&/@rawdata;
 				withCamelTitles=Replace[Cases[#, XMLElement[_, _, _]], {XMLElement["string", _, {b_}] :> Rule["ID", b], XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity]&/@rawdata;
				withCamelTitles = MapThread[List, {Rule["InChI", #] & /@ ("InChI"/.newparams), withCamelTitles}];
				Dataset[Association /@ withCamelTitles]
			)]
		)]
	),{"InChIKey"},
	(
		If[!(StringQ["InChIKey"/.newparams]||MatchQ["InChIKey"/.newparams,List[__String]]),  
		(	
			Message[ServiceExecute::nval,"InChIKey","ChemSpider"];
			Throw[$Failed]
		)];
		Switch[output,
		"InChIKey",
		(
			Message[ServiceExecute::siden];
			Throw[$Failed]
		),"MOL",
		(
			If[StringQ["InChIKey"/.newparams],
			(
				rawdata = ServiceExecute["ChemSpider","RawInChIKeyToMol",{"inchi_key"->("InChIKey"/.newparams)}];
 				withCamelTitles=StringReplace[rawdata, ___ ~~ "<string xmlns=\"http://www.chemspider.com/\">" ~~ a__ ~~ "</string>" ~~ ___ :> a];
				
				Dataset[Association["MOL"->withCamelTitles]]
			),
			(
				rawdata = ServiceExecute["ChemSpider","RawInChIKeyToMol",{"inchi_key"->#}]&/@("InChIKey"/.newparams);
 				withCamelTitles=StringReplace[#, ___ ~~ "<string xmlns=\"http://www.chemspider.com/\">" ~~ a__ ~~ "</string>" ~~ ___ :> a]&/@rawdata;
				withCamelTitles = List["MOL"->#]&/@withCamelTitles;
				withCamelTitles = MapThread[List, {Rule["InChIKey", #] & /@ ("InChIKey"/.newparams), withCamelTitles}];
				Dataset[Association/@withCamelTitles]
			)]
		),"InChI",
		(
			If[StringQ["InChIKey"/.newparams],
			(
				rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawInChIKeyToInChI",{"inchi_key"->("InChIKey"/.newparams)}],"XML"]];
				If[MatchQ[rawdata,$Failed],
   				(
      				Message[ServiceExecute::serrormsg,""];
       				Throw[$Failed]
 				)];
 				withCamelTitles=Replace[Cases[rawdata, XMLElement[_, _, _]], {XMLElement["string", _, {b_}] :> Rule["InChI", b], XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity];
				Dataset[Association[withCamelTitles]]
			),
			(
				rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawInChIKeyToInChI",{"inchi_key"->#}],"XML"]]&/@("InChIKey"/.newparams);
				If[MatchQ[#,$Failed],
   				(
      				Message[ServiceExecute::serrormsg,""];
       				Throw[$Failed]
 				)]&/@rawdata;
 				withCamelTitles=Replace[Cases[#, XMLElement[_, _, _]], {XMLElement["string", _, {b_}] :> Rule["InChI", b], XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity]&/@rawdata;
				withCamelTitles = MapThread[List, {Rule["InChIKey", #] & /@ ("InChIKey"/.newparams), withCamelTitles}];
				Dataset[Association /@ withCamelTitles]
			)]
		),"ID",
		(
			If[StringQ["InChIKey"/.newparams],
			(
				rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawInChIKeyToCSID",{"inchi_key"->("InChIKey"/.newparams)}],"XML"]];
				If[MatchQ[rawdata,$Failed],
   				(
      				Message[ServiceExecute::serrormsg,""];
       				Throw[$Failed]
 				)];
 				withCamelTitles=Replace[Cases[rawdata, XMLElement[_, _, _]], {XMLElement["string", _, {b_}] :> Rule["ID", b], XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity];
				Dataset[Association[withCamelTitles]]
			),
			(
				rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawInChIKeyToCSID",{"inchi_key"->#}],"XML"]]&/@("InChIKey"/.newparams);
				If[MatchQ[#,$Failed],
   				(
      				Message[ServiceExecute::serrormsg,""];
       				Throw[$Failed]
 				)]&/@rawdata;
 				withCamelTitles=Replace[Cases[#, XMLElement[_, _, _]], {XMLElement["string", _, {b_}] :> Rule["ID", b], XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity]&/@rawdata;
				withCamelTitles = MapThread[List, {Rule["InChIKey", #] & /@ ("InChIKey"/.newparams), withCamelTitles}];
				Dataset[Association /@ withCamelTitles]
			)]
		)]
	),{"ID"},
	(
		If[!(IntegerQ["ID"/.newparams]||MatchQ["ID"/.newparams,List[__Integer]]),
		(	
			Message[ServiceExecute::nval,"ID","ChemSpider"];
			Throw[$Failed]
		)];
		Switch[output,
		"ID",
		(
			Message[ServiceExecute::siden];
			Throw[$Failed]
		),"MOL",
		(
			If[IntegerQ["ID"/.newparams],
			(
				rawdata = ServiceExecute["ChemSpider","RawCSIDToMol",{"csid"->ToString["ID"/.newparams]}];
				withCamelTitles=StringReplace[rawdata, ___ ~~ "<string xmlns=\"http://www.chemspider.com/\">" ~~ a__ ~~ "</string>" ~~ ___ :> a];
				Dataset[Association["MOL"->withCamelTitles]]
			),
			(
				rawdata = ServiceExecute["ChemSpider","RawCSIDToMol",{"csid"->ToString[#]}]&/@("ID"/.newparams);
 				withCamelTitles=StringReplace[#, ___ ~~ "<string xmlns=\"http://www.chemspider.com/\">" ~~ a__ ~~ "</string>" ~~ ___ :> a]&/@rawdata;
				withCamelTitles = List["MOL"->#]&/@withCamelTitles;
				withCamelTitles = MapThread[List, {Rule["ID", #] & /@ ("ID"/.newparams), withCamelTitles}];
				Dataset[Association/@withCamelTitles]
			)]
		),"InChI",
		(
			If[IntegerQ["ID"/.newparams],
			(
				rawdata = ServiceExecute["ChemSpider","RawCSIDToMol",{"csid"->ToString["ID"/.newparams]}];
				rawdata=StringReplace[rawdata, ___ ~~ "<string xmlns=\"http://www.chemspider.com/\">" ~~ a__ ~~ "</string>" ~~ ___ :> a];
				rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawMolToInChI",{"mol"->rawdata}],"XML"]];
				If[MatchQ[rawdata,$Failed],
   				(
      				Message[ServiceExecute::serrormsg,""];
       				Throw[$Failed]
 				)];
 				withCamelTitles=Replace[Cases[rawdata, XMLElement[_, _, _]], {XMLElement["string", _, {b_}] :> Rule["InChI", b], XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity];
 				Dataset[Association[withCamelTitles]]
			),
			(
				rawdata = ServiceExecute["ChemSpider","RawCSIDToMol",{"csid"->ToString[#]}]&/@("ID"/.newparams);
 				rawdata=StringReplace[#, ___ ~~ "<string xmlns=\"http://www.chemspider.com/\">" ~~ a__ ~~ "</string>" ~~ ___ :> a]&/@rawdata;
				rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawMolToInChI",{"mol"->#}],"XML"]]&/@rawdata;
				If[MatchQ[#,$Failed],
   				(
      				Message[ServiceExecute::serrormsg,""];
       				Throw[$Failed]
 				)]&/@rawdata;
 				withCamelTitles=Replace[Cases[#, XMLElement[_, _, _]], {XMLElement["string", _, {b_}] :> Rule["InChI", b], XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity]&/@rawdata;
 				withCamelTitles = MapThread[List, {Rule["ID", #] & /@ ("ID"/.newparams), withCamelTitles}];
 				Dataset[Association/@withCamelTitles]
			)]
		),"InChIKey",
		(
			If[IntegerQ["ID"/.newparams],
			(
				rawdata = ServiceExecute["ChemSpider","RawCSIDToMol",{"csid"->ToString["ID"/.newparams]}];
				rawdata=StringReplace[rawdata, ___ ~~ "<string xmlns=\"http://www.chemspider.com/\">" ~~ a__ ~~ "</string>" ~~ ___ :> a];
				rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawMolToInChIKey",{"mol"->rawdata}],"XML"]];
				If[MatchQ[rawdata,$Failed],
   				(
    	  			Message[ServiceExecute::serrormsg,""];
      	 			Throw[$Failed]
 				)];
 				withCamelTitles=Replace[Cases[rawdata, XMLElement[_, _, _]], {XMLElement["string", _, {b_}] :> Rule["InChIKey", b], XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity];
 				Dataset[Association[withCamelTitles]]
			),
			(
				rawdata = ServiceExecute["ChemSpider","RawCSIDToMol",{"csid"->ToString[#]}]&/@("ID"/.newparams);
 				rawdata=StringReplace[#, ___ ~~ "<string xmlns=\"http://www.chemspider.com/\">" ~~ a__ ~~ "</string>" ~~ ___ :> a]&/@rawdata;
				rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawMolToInChIKey",{"mol"->#}],"XML"]]&/@rawdata;
				If[MatchQ[#,$Failed],
   				(
      				Message[ServiceExecute::serrormsg,""];
       				Throw[$Failed]
 				)]&/@rawdata;
 				withCamelTitles=Replace[Cases[#, XMLElement[_, _, _]], {XMLElement["string", _, {b_}] :> Rule["InChIKey", b], XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity]&/@rawdata;
 				withCamelTitles = MapThread[List, {Rule["ID", #] & /@ ("ID"/.newparams), withCamelTitles}];
 				Dataset[Association/@withCamelTitles]
			)]
		)]
	),{},
	(
		Message[ServiceExecute::nparam,"MOL, InChI, InChIKey or ID","ChemSpider"];
		Throw[$Failed]
	)]
]



(*chemspidercookeddata["MOLToInChI", id_,args_]:=Block[{rawdata,newparams,params={},invalidParameters,withCamelTitles},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"MOL"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"ChemSpider"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"MOL"],
	(
		If[!StringQ["MOL"/.newparams],
		(	
			Message[ServiceExecute::nval,"MOL","ChemSpider"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["mol",ToString["MOL"/.newparams]]]
	),
	(
		Message[ServiceExecute::nparam,"MOL","ChemSpider"];
		Throw[$Failed]
	)];
	rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawMolToInChI",params],"XML"]];
	If[MatchQ[rawdata,$Failed],
   	(
      	Message[ServiceExecute::serrormsg,""];
       	Throw[$Failed]
 	)];
 	withCamelTitles=Replace[Cases[rawdata, XMLElement[_, _, _]], {XMLElement["string", _, {b_}] :> Rule["InChI", b], XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity];
	Dataset[Association[withCamelTitles]]
]

chemspidercookeddata["MOLToInChIKey", id_,args_]:=Block[{rawdata,newparams,params={},invalidParameters,withCamelTitles},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"MOL"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"ChemSpider"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"MOL"],
	(
		If[!StringQ["MOL"/.newparams],
		(	
			Message[ServiceExecute::nval,"MOL","ChemSpider"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["mol",ToString["MOL"/.newparams]]]
	),
	(
		Message[ServiceExecute::nparam,"MOL","ChemSpider"];
		Throw[$Failed]
	)];
	rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawMolToInChIKey",params],"XML"]];
	If[MatchQ[rawdata,$Failed],
   	(
      	Message[ServiceExecute::serrormsg,""];
       	Throw[$Failed]
 	)];
 	withCamelTitles=Replace[Cases[rawdata, XMLElement[_, _, _]], {XMLElement["string", _, {b_}] :> Rule["InChIKey", b], XMLElement[a_String, _, b_] :> Rule[a, b]}, Infinity];
	Dataset[Association[withCamelTitles]]
]


chemspidercookeddata["MOLToID", id_,args_]:=Block[{rawdata,newparams,params={},invalidParameters,withCamelTitles},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"MOL","Options"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"ChemSpider"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"MOL"],
	(
		If[!StringQ["MOL"/.newparams],
		(	
			Message[ServiceExecute::nval,"MOL","ChemSpider"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["mol","MOL"/.newparams]]
	),
	(
		Message[ServiceExecute::nparam,"MOL","ChemSpider"];
		Throw[$Failed]
	)];
  	If[KeyExistsQ[newparams,"Options"],
	(
		If[!StringMatchQ[ToString["Options" /. newparams],"ExactMatch"|"AllTautomers"|"SameSkeletonAndH"|"SameSkeleton"|"AllIsomers"],
		(	
			Message[ServiceExecute::nval,"Options","ChemSpider"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["options",(ToString["Options" /. newparams]/.{"ExactMatch"->"eExactMatch","AllTautomers"->"eAllTautomers","SameSkeletonAndH"->"eSameSkeletonAndH","SameSkeleton"->"eSameSkeleton","AllIsomers"->"eAllIsomers"})]]
	),
  	(
  		params = Append[params,Rule["options","eExactMatch"]]
  	)];
 	rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawMol2CSID",params],"XML"]];
	If[MatchQ[rawdata,$Failed],
   	(
      	Message[ServiceExecute::serrormsg,""];
       	Throw[$Failed]
 	)];
 	withCamelTitles=("ArrayOfInt" /. Replace[Cases[rawdata, XMLElement[_, _, _]], {XMLElement[a_String, _, b_] :>  Rule[a, b]}, Infinity]) /. {Rule["int", {b_}] :> Rule["ID", b]};
	Dataset[Association /@ withCamelTitles]
]

chemspidercookeddata["RecordToMOL", id_,args_]:=Block[{rawdata,newparams,params={},invalidParameters,withCamelTitles},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"ID","Calc3D"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"ChemSpider"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"ID"],
	(
		If[!(StringQ["ID"/.newparams]||IntegerQ["ID"/.newparams]),
		(	
			Message[ServiceExecute::nval,"ID","ChemSpider"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["csid",ToString["ID"/.newparams]]]
	),
	(
		Message[ServiceExecute::nparam,"ID","ChemSpider"];
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"Calc3D"],
	(
		If[!MemberQ[{True,False},"Calc3D"/.newparams],
		(	
			Message[ServiceExecute::nval,"Calc3D","ChemSpider"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["calc3d",("Calc3D"/.newparams)/.{True -> "true", False -> "false"}]]
	),
	(
		params = Append[params,Rule["calc3d","false"]]
	)];
	rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawGetRecordMOL",params],"XML"]];
	If[MatchQ[rawdata,$Failed],
   	(
      	Message[ServiceExecute::serrormsg,""];
       	Throw[$Failed]
 	)];
	withCamelTitles=Replace[Cases[rawdata, XMLElement[_, _, _]], {XMLElement[a_String, _, b_] :> Rule[a, b]},Infinity] /. {Rule[c_, {d_}] :> Rule[c, d]} /. {Rule["string", e_] :> Rule["MOL", e]};
	Dataset[Association[withCamelTitles]]
]

chemspidercookeddata["IDToMOL", id_,args_]:=Block[{rawdata,newparams,params={},invalidParameters,withCamelTitles},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"ID"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"ChemSpider"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"ID"],
	(
		If[!(StringQ["ID"/.newparams]||IntegerQ["ID"/.newparams]),
		(	
			Message[ServiceExecute::nval,"ID","ChemSpider"];
			Throw[$Failed]
		)];
		params = Append[params,Rule["csid",ToString["ID"/.newparams]]]
	),
	(
		Message[ServiceExecute::nparam,"ID","ChemSpider"];
		Throw[$Failed]
	)];
	rawdata = Quiet[ImportString[ServiceExecute["ChemSpider","RawCSIDToMol",params],"XML"]];
	If[MatchQ[rawdata,$Failed],
   	(
      	Message[ServiceExecute::serrormsg,""];
       	Throw[$Failed]
 	)];
	withCamelTitles=Replace[Cases[rawdata, XMLElement[_, _, _]], {XMLElement[a_String, _, b_] :> Rule[a, b]},Infinity] /. {Rule[c_, {d_}] :> Rule[c, d]} /. {Rule["string", e_] :> Rule["MOL", e]};
	Dataset[Association[withCamelTitles]]
]*)

chemspidercookeddata[___]:=$Failed

chemspidersendmessage[___]:=$Failed

End[]

End[]

SetAttributes[{},{ReadProtected, Protected}];


(* Return three functions to define oauthservicedata, oauthcookeddata, oauthsendmessage  *)
{ChemSpiderAPI`Private`chemspiderdata,ChemSpiderAPI`Private`chemspidercookeddata,ChemSpiderAPI`Private`chemspidersendmessage}
