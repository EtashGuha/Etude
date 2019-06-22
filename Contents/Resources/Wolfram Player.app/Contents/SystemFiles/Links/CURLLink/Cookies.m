(* Wolfram Language Package *)

BeginPackage["CURLLink`Cookies`"]
(* Exported symbols added here with SymbolName::usage *)  
System`$Cookies
Developer`$SessionCookies
System`SetCookies
System`ClearCookies
System`FindCookies
System`$CookieStore
Developer`$PersistentCookies
GetKnownCookies;
CookieStore;
SetAttributes[System`$Cookies,{Protected}]
Begin["`Private`"] (* Begin Private Context *) 
Needs["CURLLink`"]
Needs["CURLLink`HTTP`"]

(*
Set the default directory to be .Cookies on OSX and Unix
systems s.t cookie are stored in a hidden folder. 
On Windows, a file name cannot begin with a ".".
*)
defaultCookieDir:=defaultCookieDir=Which[
	MemberQ[{"MacOSX","Unix"},$OperatingSystem],
	FileNameJoin[{$UserBaseDirectory,".cookies"}]
	,
	$OperatingSystem==="Windows",
	FileNameJoin[{$UserBaseDirectory,"cookies"}]
]
defaultCookieFile:=defaultCookieFile=FileNameJoin[{defaultCookieDir,"cookies"}];

Unprotect[System`$Cookies]

Experimental`SetDelayedSpecialSymbol[System`$Cookies, {}];
SetAttributes[cookieAccessor, HoldAll];
Experimental`DelayedValueFunction[System`$Cookies] = cookieAccessor;
cookieAccessor[ sym_] := (cookiesToAssociation[GetKnownCookies[]]);
cookieAccessor[sym_, val_] := (System`SetCookies[val] ;True)

getCookieStore[file_]:=
	Check[
		If[Quiet[CURLLink`HTTP`Private`isFile[file]],
			(*then*)
			Quiet[
				Check[
					(*create the file and return file object*)
					PutAppend[file];
					File[CURLLink`HTTP`Private`getFile[file]]
					,
					(*else issue error and set $CookieStore to None*)
					Message[$CookieStore::noopen,file];
					getCookieStore[None]]
					,
				PutAppend::noopen]
				,
			(*
			we have an invalid file name,
			set $CookieStore to None
			*)
			Message[$CookieStore::invfile,file];
			getCookieStore[None]]
			,
			getCookieStore[None]
     	 ]

getDefaultCookieStore[Automatic]:=
Catch[
	If[FileExistsQ@defaultCookieFile,
		(*then, we're done--return defaultCookieFile*)
		File[defaultCookieFile]
		,
		(*else*)
		If[DirectoryQ[defaultCookieDir],
			(*
			if defaultCookieDir exists: create,
			or append-to defaultCookieFile
			*)
			PutAppend[File[defaultCookieFile]];
			File[defaultCookieFile]
			,
			(*else*)
			Quiet[
				Check[
					(*create a defaultCookieDir*)
					CreateDirectory[defaultCookieDir]
					,
					(*
					If we can't create defaultCookieDir,
					issue an error and set $CookieStore
					to None.
					*)
					Message[$CookieStore::filex,defaultCookieFile];
					Throw[getCookieStore[None],CURLLink`Utilities`Exception]
					],
				CreateDirectory::dirf];
			(*
			Upon successful creation of defaultCookieDir--
			create, or append-to defaultCookieFile
			*)
			Quiet[
				Check[
				PutAppend[File[defaultCookieFile]];
				File[defaultCookieFile]
				,
				(*
				if defaultCookieFile can't be
				created, or appended-to, set 
				$CookieStore to None
				*)
				Message[$CookieStore::filex,defaultCookieFile];
				Throw[getCookieStore[None],CURLLink`Utilities`Exception]
				]]
			]		
	  ],CURLLink`Utilities`Exception]
	
getCookieStore[None,CURLLink`Utilities`Exception]:=None
getCookieStore[None]:=None
getCookieStore[Automatic]:=If[FileExistsQ[defaultCookieFile],File[defaultCookieFile],None]
(*
Set Delayed special symbol $CookieStore and
set it's default value to None
*)
Experimental`SetDelayedSpecialSymbol[System`$CookieStore, None];
SetAttributes[cookieStoreAccessor, HoldAll];
Experimental`DelayedValueFunction[System`$CookieStore]= cookieStoreAccessor;
cookieStoreAccessor[ sym_]:= getCookieStore[sym]
cookieStoreAccessor[sym_, Automatic]:=Quiet[Check[FileExistsQ[getDefaultCookieStore[Automatic]],False],FileExistsQ::fstr]
cookieStoreAccessor[sym_, None]:= True
cookieStoreAccessor[sym_, val_]:=Quiet[Check[FileExistsQ[getCookieStore[val]],False],FileExistsQ::fstr] 


$CookieStore::invfile="`1` is not a valid File";
$CookieStore::noopen="Cannot open `1`"
$CookieStore::filex="Cannot write to file `1`."
Protect[System`$Cookies]


(*
LoadPersistentCookies:
This is being called in HTTP.m 
at initialization
*)
LoadPersistentCookies[store_]:=
Module[
	{handle},
	handle = CURLLink`CURLHandleLoad[];
	CURLLink`HTTP`Private`setStandardOptions[handle, ""];
	handle["Return"] = 0;
	CURLLink`CURLOption[handle,"CURLOPT_COOKIEFILE", Quiet@Check[CURLLink`HTTP`Private`getFile@store,""]];
 	CURLLink`CURLOption[handle,"CURLOPT_COOKIELIST", "RELOAD"];
 	CURLLink`CURLOption[handle,"CURLOPT_COOKIEJAR",""];
	CURLLink`CURLHandleUnload[handle];
]


(*
Find Cookies
*)
System`FindCookies[]:=System`$Cookies
System`FindCookies[parameters__]:=implFindCookies[parameters]
implFindCookies[All]:=System`$Cookies
implFindCookies[domain_String]:=implFindCookies[Association["Domain"->domain]]
implFindCookies[Rule[key_,valpattern_]]:=implFindCookies[Association[key->valpattern]]
implFindCookies[parameters_Association]:=
Module[
	{val,cookies},
	cookies=cookiesToAssociation[GetKnownCookies[]];
	Function[{assoc},
		If[AllTrue[Function[{rule},
      	Or @@ Map[(	MatchQ[Lookup[assoc, Keys[rule]], #]
          			 ||
          			If[StringQ[#] && StringQ[(val=Lookup[assoc, Keys[rule]])],StringMatchQ[val, #],False,False]
          			 ||
          			If[DateObjectQ[#],Round[DateDifference[Lookup[assoc, Keys[rule]], #,"Day"]]===Quantity[0,"Days"],False,False]
          		  ) &, Flatten@{Values[rule]}]
          		  ] /@ Flatten@{Normal[parameters]}, TrueQ], assoc, 
   Nothing]] /@ cookies
   ]


(*Clear Cookies*)
System`ClearCookies[args__]:=With[{res = implClearCookies[args]}, res /; res =!= $Failed]

implClearCookies[All]:=
Module[{handle, error,cookiesToBeDeleted},
		error = Catch[
			cookiesToBeDeleted=$Cookies;
			handle = CURLLink`CURLHandleLoad[];
			CURLLink`CURLAutoCookies[handle];
			Quiet[Check[Put[CURLLink`HTTP`Private`getFile@System`$CookieStore],Null,DeleteFile::fdnfnd]];
			(*ask libcurl to use $CookieStore to write cookies to *)
			CURLLink`CURLOption[handle,"CURLOPT_COOKIEJAR",Quiet@Check[CURLLink`HTTP`Private`getFile@System`$CookieStore,""]];
			
			(*ask libcurl to use $CookieStore to read cookies from*)
			CURLLink`CURLOption[handle,"CURLOPT_COOKIEFILE", Quiet@Check[CURLLink`HTTP`Private`getFile@System`$CookieStore,""]];
			
			(*Clear all the cookies in memory*)
			CURLLink`CURLOption[handle,"CURLOPT_COOKIELIST","ALL"];
			
			handle["Return"] = 0;
			CURLLink`HTTP`Private`$LastKnownCookies="";
			Unprotect[System`$Cookies];
			System`$Cookies={};
			Protect[System`$Cookies];
			CURLLink`CURLHandleUnload[handle];
		,CURLLink`Utilities`Exception];
		If[error === $Failed, $Failed, cookiesToBeDeleted]
	]

implClearCookies[domain_String]:=implClearCookies[Association["Domain"->domain]];
implClearCookies[Rule[key_,value_]]:=implClearCookies[Association[key->value]];
implClearCookies[parameters_Association]:=
Module[{cookies,cookiesA, handle, error,cookiesToBeDeleted,deletedcookies,failedToDeleteCookies},
		error = Catch[
			(*find cookies to be deleted*)
			cookiesToBeDeleted=System`FindCookies[parameters];
			
			(*get all the cookies in memory*)
			cookiesA=cookiesToAssociation[GetKnownCookies[]];
			
			(*get cookies to be restored*)
			cookies=Complement[cookiesA,cookiesToBeDeleted];
			
			handle = CURLLink`CURLHandleLoad[];
			
			CURLLink`HTTP`Private`setStandardOptions[handle, ""];
			
			(*Empty file containing persistent cookies*)
			Quiet[Check[Put[CURLLink`HTTP`Private`getFile@System`$CookieStore],Null]];
			CURLLink`CURLOption[handle,"CURLOPT_COOKIEFILE", Quiet@Check[CURLLink`HTTP`Private`getFile@System`$CookieStore,""]];
			(*Clear all cookies from memory*)
			CURLLink`CURLOption[handle,"CURLOPT_COOKIELIST","ALL"];
			
			handle["Return"] = 0;
			CURLLink`CURLHandleUnload[handle];
			(*Restore cookies*)
			System`SetCookies[cookies];
			
			failedToDeleteCookies= Intersection[cookiesToAssociation[GetKnownCookies[]],cookiesToBeDeleted];
			
			(*Return cookies that did get deleted*)
			deletedcookies=Complement[cookiesToBeDeleted,failedToDeleteCookies];
			
			
			(*issue a message if some cookies did not get deleted *)
			If[Complement[cookiesToBeDeleted,deletedcookies]=!={},$Failed,deletedcookies]
		,CURLLink`Utilities`Exception];
		If[error === $Failed, $Failed, deletedcookies]
	]

implClearCookies[args___] :=(System`Private`Arguments[System`ClearCookies[args], 1];$Failed)
(*
Set Cookies:
-This _adds_ cookies to the cookie-jar if they're persistent
-Updates $Cookies
-Update libcurls knowledge of cookies
-libcurl looses the knowledge of cookie-jar
*)
System`SetCookies[args__]:=With[{res = implSetCookies[args]}, res /; res =!= $Failed]
implSetCookies[ucookie_]:=
Module[{handle, error,cookiesStringList},
		error = Catch[
					handle = CURLLink`CURLHandleLoad[];
					CURLLink`HTTP`Private`setStandardOptions[handle, ""];
					(*this bit coverts user input into netscape style cookie string*)
					cookiesStringList=
							Which[
								TrueQ[listOfAssocQ[ucookie]],
								Map[cookieFromAssociation,ucookie]
								,
								AssociationQ[ucookie],
								{cookieFromAssociation[ucookie]}
								,
								True,
								Throw[$Failed,CURLLink`Utilities`Exception]
								
								];
					(*add cookies to libcurls cookie engine*)
					Map[CURLLink`CURLOption[handle,"CURLOPT_COOKIELIST",#]&,cookiesStringList];
					
					(*save to cookie-store if $CookieStore is set to a valid file*)
					writeCookiesToCookieStore[handle];
					
					handle["Return"] = 0;
					CURLLink`CURLHandleUnload[handle];
					,
					CURLLink`Utilities`Exception];
		
		If[error === $Failed, $Failed, ucookie]
	]
implSetCookies[args___] :=(System`Private`Arguments[System`SetCookies[args], 1];$Failed)

writeCookiesToCookieStore[handle_]:=
Module[
	{allcookies,fileExistsQ},
	fileExistsQ=Quiet[TrueQ[FileExistsQ[$CookieStore]]];
	If[fileExistsQ,
		(*get all the cookies, we'll need to restore them*)
		allcookies=cookiesToAssociation@CURLLink`CURLCookies[handle];
		(*Clear out the session cookies*)
		CURLLink`CURLOption[handle,"CURLOPT_COOKIELIST","SESS"];
						
		(*set the location of the cookie-jar*)
		CURLLink`CURLOption[handle,"CURLOPT_COOKIEJAR",Quiet@Check[CURLLink`HTTP`Private`getFile@System`$CookieStore,""]];
						
		(*
		put the cookies in to the cookie jar, 
		because session cookies were cleared
		they don't go to the cookie-jar
		*)
		CURLLink`CURLOption[handle,"CURLOPT_COOKIELIST","FLUSH"];
		
		(*
		make sure cookie-jar is set to "",else 
		at the CURLLink`CURLHandleUnload[] call
		libcurl will flush all the cookies, even
		the session cookies  to the cookie-jar
		*)
		CURLLink`CURLOption[handle,"CURLOPT_COOKIEJAR",""];
		
		(*We have now cleared session cookies and 
		flushed persistent cookies, so libcurl has
		no cookies, restore them*)
		Map[CURLLink`CURLOption[handle,"CURLOPT_COOKIELIST",cookieFromAssociation[#]]&,allcookies];
	  ]
]	
GetKnownCookies[]/;CURLLink`HTTP`Private`initializeQ[]:=
Module[{handle,cookiestring},
	Quiet[If[FileExistsQ[$CookieStore],LoadPersistentCookies[$CookieStore]]];
	handle = CURLLink`CURLHandleLoad[];
	CURLLink`HTTP`Private`setStandardOptions[handle, ""];
	
	(*Must set CURLOPT_COOKIEJAR option to "",
	else libcurl will write session cookies
	to $CookieStore when CURLLink`CURLHandleUnload
	is called*)
	CURLLink`CURLOption[handle,"CURLOPT_COOKIEJAR",""];
	handle["Return"] = 0;
	cookiestring=CURLLink`CURLCookies[handle];
	CURLLink`CURLHandleUnload[handle];
	cookiestring	
]

(*convert cookies to association*)
listOfAssocQ[list_List]:=AllTrue[Map[AssociationQ[#]&,list],TrueQ]

listOfStringQ[list_List]:=AllTrue[Map[StringQ[#]&,list],TrueQ]

cookiesToAssociation[cookieString_String]:=
StringCases[cookieString,Shortest[domain__ ~~ "\t" ~~ allowSubdomains__ ~~ "\t" ~~ path__ ~~ "\t" ~~ secureFlag__ ~~ "\t" ~~ utime__ ~~ "\t" ~~ name__ ~~ "\t" ~~ value__ ~~ "\n"] :>
  <|
   "Domain" ->(dropLeadingDot[StringDelete[domain,"#HttpOnly_"]]),
   "Path" -> path,
   "Name" -> name,
   "Content" -> value,
   "ExpirationDate" ->If[StringMatchQ[utime,"0"], Automatic,FromUnixTime[ToExpression[utime]]],
   "AllowSubdomains"->ToLowerCase[allowSubdomains] /. {"true" -> True,"false" -> False},
   "ConnectionType" -> ToLowerCase[secureFlag] /. {"true" -> "HTTPS","false" -> All},
   "ScriptAccessible"->If[StringContainsQ[domain,"httponly",IgnoreCase->True],False,True]
   |>]

dropLeadingDot[string_String] :=StringReplace[string, RegularExpression["^\\."]->""]

cookieFromAssociation[assoc_Association]:=
Module[
	{t,allowsubdomains},
	allowsubdomains=StringMatchQ[Lookup[assoc,"Domain",""],"."~~__];
	StringJoin[{
		If[(Lookup[assoc,"ScriptAccessible",""])===False,"#HttpOnly_",""],
		If[Lookup[assoc,"AllowSubdomains",False],".",""],
		Lookup[assoc,"Domain","."],"\t",
		ToUpperCase[ToString[If[Lookup[assoc,"AllowSubdomains",allowsubdomains]===True,"TRUE","FALSE"]]],"\t",
		Lookup[assoc,"Path","/"],"\t",
		ToUpperCase[ToString[If[Lookup[assoc,"ConnectionType",""]===All,"FALSE","TRUE"]]],"\t",
		If[(t=Lookup[assoc,"ExpirationDate",Automatic])===Automatic,"0",ToString[UnixTime[t]]],"\t",
		Lookup[assoc,"Name",""],"\t",
		Lookup[assoc,"Content",Lookup[assoc,"Value",""]],
		"\n"
				
		}]

]

Developer`$PersistentCookies/;CURLLink`HTTP`Private`initializeQ[] :=
	Module[{cookies, handle, error},
		error = Catch[
			handle = CURLLink`CURLHandleLoad[];
			CURLLink`HTTP`Private`setStandardOptions[handle, ""];
			handle["Return"] = 0;
			CURLLink`CURLOption[handle,"CURLOPT_COOKIELIST", "SESS"];
			cookies = cookiesToAssociation[CURLLink`CURLCookies[handle]];
			CURLLink`CURLHandleUnload[handle];
		,CURLLink`Utilities`Exception];
		If[error === $Failed, $Failed, cookies]
	]
(*Convert new cookie to old one for backwards compatibility*)	
toOldCookie[newCookie_]:= KeyValueMap[transform,newCookie]

transform["ExpirationDate",date_DateObject]:="Expires"->DateString[date]
transform["ExpirationDate",d_String]:="Expires"->d
transform["ExpirationDate",_]:="Expires"->"Thu 1 Jan 1970 00:00:00"
transform["ConnectionType","HTTPS"]:="Secure"->"TRUE"
transform["ConnectionType",_]:="Secure"->"FALSE"
transform["Content",cont_String]:="Value"->cont
transform["Content",cont_]:="Value"->ToString[cont]
transform["AllowSubdomains",val_?BooleanQ]:="MachinceAccess"->(val/.{True->"TRUE",False->"FALSE"})

transform[key_,val_]:=key->val


End[] (* End Private Context *)
EndPackage[]

