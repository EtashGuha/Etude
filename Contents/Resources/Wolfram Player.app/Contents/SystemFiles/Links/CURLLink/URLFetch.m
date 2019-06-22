(* Wolfram Language Package *)

BeginPackage["CURLLink`URLFetch`"]
(* Exported symbols added here with SymbolName::usage *)
URLFetch::usage = "URLFetch[url, elements] return elements from url, for any accessible URL.";
URLSave::usage = "URLSave[url, file, elements] return elements from url for any accessible URL, and store the content in file. ";
URLFetchAsynchronous::usage = "URLFetchAsynchronous[url, eventFunction] asynchronously connect to a URL";
URLSaveAsynchronous::usage = "URLSaveAsynchronous[url, file, eventFunction] asynchronously connect to a URL, and store the content in a file.";
$HTTPCookies::usage = "Returns the list of globally shared cookies."
SetAttributes[URLFetch, {ReadProtected}];
SetAttributes[URLSave, {ReadProtected}];
SetAttributes[URLFetchAsynchronous, {ReadProtected}];
SetAttributes[URLSaveAsynchronous, {ReadProtected}];

Begin["`Private`"] (* Begin Private Context *) 

Needs["CURLLink`"]
Needs["PacletManager`"]
Needs["OAuthSigning`"]
Needs["CURLLink`HTTP`"]


If[$VersionNumber <  9,
	Message[CURLLink::enable,  "CURLLink"]
]

$Handlers = <||>


CURLLink`Utilities`AddHandler[ name_, assoc_] :=
	Module[ {},
		(*
		 Should verify the contents of the association.
		*)
		AssociateTo[ $Handlers, name -> assoc]
	]
	

LookupHandler[ name_, funName_] :=
	Module[ {funs, fun},
		Catch[
				funs = Lookup[ $Handlers, name, Null];
				If[ funs === Null, 
					Throw[$Failed,CURLLink`Utilities`Exception]];
				fun = Lookup[ funs, funName, Null];
				If[ fun === Null, 
					Throw[$Failed,CURLLink`Utilities`Exception]];
				fun
				,
				CURLLink`Utilities`Exception
			 ]
	]
Options[URLFetch]= 
{	
	Method -> "GET", 
	"Parameters" -> {},
	"Body" -> "", 
	"MultipartElements" -> {},
	"VerifyPeer" -> True, 
	"Username" -> None, 
	"Password" -> None, 
	"UserAgent" -> Automatic, 
	System`CookieFunction->Automatic,
	"Cookies"->Automatic, 
	"StoreCookies" -> True,
	"Headers" -> {},
	"CredentialsProvider"->Automatic,
	"ConnectTimeout"->0,
	"ReadTimeout"->0,
	"DisplayProxyDialog" -> True,
	"OAuthAuthentication" -> None,
	"FollowRedirects" -> True,
	"ProxyUsername"->"",
	"ProxyPassword"->"",
	"Debug"->False,
	ConnectionSettings-><|"MaxUploadSpeed"->Automatic,"MaxDownloadSpeed"->Automatic|>
}
Options[URLSave] = Join[Options[URLFetch], {BinaryFormat->True}]
Options[URLFetchAsynchronous] = Join[FilterRules[Options[URLFetch],Except["DisplayProxyDialog"]], {"DisplayProxyDialog"->False,"Progress"->False, "Transfer"->Automatic, "UserData"->None,"Events"->Automatic}];
Options[URLSaveAsynchronous] = Join[FilterRules[Options[URLFetch],Except["DisplayProxyDialog"]], {"DisplayProxyDialog"->False,"Progress"->False, BinaryFormat->True, "UserData"->None,"Events"->Automatic}]

URLFetch[request__]:=
	With[
		{
			res=
			Catch[
			LookupHandler["http","URLFetch"][request],
			CURLLink`Utilities`Exception,
			CURLLink`Utilities`errorHandler
			]
		},
		res/;res=!=False
		]

URLFetchAsynchronous[request__]:=
	With[
		{
			res=
			Catch[
			LookupHandler["http","URLFetchAsynchronous"][request],
			CURLLink`Utilities`Exception,
			CURLLink`Utilities`errorHandler
			]
		},
		res/;res=!=False
		]

URLSave[request__]:=
	With[
		{
			res=
			Catch[	
			LookupHandler["http","URLSave"][request],
			CURLLink`Utilities`Exception,
			CURLLink`Utilities`errorHandler
			]
		},
		res/;res=!=False
		]

URLSaveAsynchronous[request__]:=
	With[
		{
			res=
			Catch[
				LookupHandler["http","URLSaveAsynchronous"][request],
				CURLLink`Utilities`Exception,
				CURLLink`Utilities`errorHandler
				]
		},
		res/;res=!=False
		]

End[] (* End Private Context *)

EndPackage[]