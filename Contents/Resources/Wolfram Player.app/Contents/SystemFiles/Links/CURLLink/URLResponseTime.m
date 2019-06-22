(* Wolfram Language Package *)

BeginPackage["CURLLink`URLResponseTime`"]
(* Exported symbols added here with SymbolName::usage *)  
System`URLResponseTime
Begin["`Private`"] (* Begin Private Context *)
SetAttributes[System`URLResponseTime, {ReadProtected}];
Needs["CURLLink`Utilities`"]


Options[System`URLResponseTime]={TimeConstraint->60,"Username"->"","Password"->"","FollowRedirects"->True,VerifySecurityCertificates->True}

System`URLResponseTime::tout = "`1`.";
System`URLResponseTime::general="`1`.";
System`URLResponseTime::invurl="`1` is not a valid URL";


toWLName=
<|
	"CURLINFO_TOTAL_TIME"			-> "TransactionTotal",
	"CURLINFO_NAMELOOKUP_TIME"		-> "NameLookup",
	"CURLINFO_CONNECT_TIME"			-> "HostConnection",
	"CURLINFO_PRETRANSFER_TIME"		-> "PreTransfer",
	"CURLINFO_STARTTRANSFER_TIME"	-> "TransferInitiation",
	"CURLINFO_REDIRECT_TIME"		-> "HTTPRedirect",
	"CURLINFO_APPCONNECT_TIME"		-> "SSLHandshake"
|>
ordering=
<|
	"NameLookup"		->	1,
	"HostConnection"	->	2,
	"SSLHandshake"		->	3,
	"PreTransfer"		->	4,
	"TransferInitiation"->	5,
	"TransactionTotal"	->	6,
	"HTTPRedirect"		->	7
|>
elements=Keys[toWLName];

System`URLResponseTime[url_,opts:OptionsPattern[]]:=(
With[
	{res=
		Catch[
			implURLResponseTime[url,"TransactionTotal",opts]
			,
			CURLLink`Utilities`Exception
			,
			errorHandler
			]
	},
	res /; res=!=False 
	]
)

System`URLResponseTime[url_,elements_,opts:OptionsPattern[]]:=(
With[
	{res=
		Catch[
			implURLResponseTime[url,elements,opts]
			,
			CURLLink`Utilities`Exception
			,
			errorHandler
			]
	},
	res /; res=!=False 
	]
)

System`URLResponseTime[args___]:=(
With[
	{res=
		Catch[
			implURLResponseTime[args]
			,
			CURLLink`Utilities`Exception
			,
			errorHandler[#1,#2,Length[Flatten@{args}]]&
			]
	},
	res /; res=!=False
	]
)

implURLResponseTime[url_,element_String,opts:OptionsPattern[System`URLResponseTime]]:=
Catch[
	Lookup[implURLResponseTime[url,All,opts],element]	,
	CURLLink`Utilities`Exception
	,
	errorHandler
	]

implURLResponseTime[url_,elements_List,opts:OptionsPattern[System`URLResponseTime]]:=
Catch[KeyTake[implURLResponseTime[url,All,opts],elements]	,
		CURLLink`Utilities`Exception
		,
		errorHandler
	] 

implURLResponseTime[url_,"ConnectionTimes",opts:OptionsPattern[System`URLResponseTime]]:=
Catch[KeyTake[implURLResponseTime[url,All,opts],{"HostConnection","SSLHandshake"}]
	,
		CURLLink`Utilities`Exception
		,
		errorHandler
	] 
 

implURLResponseTime[urlExp_/;CURLLink`Utilities`isURL[urlExp,System`URLResponseTime],All,opts:OptionsPattern[System`URLResponseTime]]:=
Module[
	{url,handle,result,err,timeout},
	
		url = CURLLink`Utilities`getURL[urlExp];
		
		timeout=CURLLink`Utilities`toIntegerMilliseconds[OptionValue[TimeConstraint]];
		
		handle = CURLLink`CURLHandleLoad[];
		
		CURLLink`CURLOption[handle, "CURLOPT_USERNAME", OptionValue["Username"]];
		
		CURLLink`CURLOption[handle, "CURLOPT_PASSWORD", OptionValue["Password"]];
		
		CURLLink`CURLOption[handle, "CURLOPT_VERIFYHOST",OptionValue[VerifySecurityCertificates]];
		
		CURLLink`CURLOption[handle, "CURLOPT_VERIFYPEER", OptionValue[VerifySecurityCertificates]];
		
		CURLLink`CURLOption[handle, "CURLOPT_USERAGENT", "Wolfram "<>ToString[$VersionNumber]];
		
		CURLLink`CURLOption[handle, "CURLOPT_TIMEOUT_MS", timeout];
		
		CURLLink`CURLOption[handle, "CURLOPT_FOLLOWLOCATION", OptionValue["FollowRedirects"]];
		
		CURLLink`CURLOption[handle, "CURLOPT_WRITEFUNCTION", "WRITE_MEMORY"];
		
		CURLLink`CURLOption[handle, "CURLOPT_WRITEDATA", "MemoryPointer"];
		
		CURLLink`CURLOption[handle, "CURLOPT_RANGE", "0-1"];
		
		CURLLink`CURLOption[handle, "CURLOPT_URL", url];
		
		If[$SystemID=!="MacOSX-x86-64",
			CURLLink`CURLOption[handle, "CURLOPT_CAINFO",CURLLink`Utilities`$CACERT];
		];
		
		CURLLink`CURLMultiHandleAdd[handle, {handle}];
		
		err=CURLLink`CURLPerform[handle, "Blocking" -> True];
		
		If[err=!=0,Throw[err,CURLLink`Utilities`Exception]];
		
		result = Association[Map[#->CURLLink`CURLGetInfo[handle, #]&,elements]];
		
		result = makeprettyResult[result];
		
		CURLLink`CURLHandleUnload[handle];
		
		result
	

]
implURLResponseTime[args_,opts:OptionsPattern[System`URLResponseTime]]:=Throw["invurl",CURLLink`Utilities`Exception]
implURLResponseTime[args___]:=Throw[False,CURLLink`Utilities`Exception]


errorHandler[28,tag_]:=(Message[System`URLResponseTime::tout,CURLLink`CURLError[28]];$Failed)

errorHandler[code_Integer,tag_]:=(Message[System`URLResponseTime::general,CURLLink`CURLError[code]];$Failed)

errorHandler[False,tag_]:=(False)

errorHandler[code_,tag_,num_Integer]:=Which[
	num>2,
	Message[System`URLResponseTime::argt, System`URLResponseTime, num,1,2];
	False
	,
	code==="invurl",
	False
	,
	True,
	Message[System`URLResponseTime::args, System`URLResponseTime];
	False
]
makeprettyResult[assoc_]:=
Module[
	{result},
	result = Map[toQuantity,assoc];
	result = KeyMap[toWLName,result];
	KeySortBy[result,ordering]
]
toQuantity=Function[{value},If[value==0.,Missing,Quantity[value,"Seconds"]]]
End[] (* End Private Context *)

EndPackage[]