(* Wolfram Language Package *)

BeginPackage["CURLLink`Utilities`"]
(* Exported symbols added here with SymbolName::usage *)  
isURL
getURL
isFile
getFile
$CACERT
toIntegerMilliseconds 
errorHandler
Begin["`Private`"] (* Begin Private Context *)
$CACERT = FileNameJoin[{DirectoryName[System`Private`$InputFileName], "SSL", "cacert.pem"}]; 
isURL[url_,head_:URLFetch]:=StringQ[getURL[url,head]];
getURL[url_String,head_:URLFetch]:=url
getURL[URL[url_],head_:URLFetch]:=getURL[url,head]
getURL[IPAddress[url_],head_:URLFetch]:=getURL[url,head]
getURL[exp_,head_:URLFetch]:=(Message[head::invurl, exp];$Failed)

isFile[exp_,head_:URLSave]:=StringQ[getFile[exp,head]];
getFile[file_String,head_:URLSave]:=file
getFile[File[file_],head_:URLSave]:=getFile[file,head]
getFile[exp_,head_:URLSave]:=(Message[head::invfile, exp];$Failed)

(*
**********************************************************
Function: 
toRealSeconds

Description:
Converts a non negative Quantity, in 
units of time, to its magnitude in seconds.
Eg. 
In[9]:= toRealSeconds[Quantity[1002, "Milliseconds"]]
Out[9]= 1.002
Note:
If no units are provided, the it is
assumed to be in seconds.
Eg: 
In[17]:= CURLLink`Utilities`toRealSeconds[1002]
Out[17]= 1002.
**********************************************************
*)
toRealSeconds[q_/;QuantityQ[q]]:=
	toRealSeconds[
		QuantityMagnitude[Check[UnitConvert[q,"Seconds"],Throw[$Failed,CURLLink`Utilities`Exception]]]
		]
toRealSeconds[q_/;NonNegative[q]]:=N[q]
toRealSeconds[q___]:=Throw[$Failed,CURLLink`Utilities`Exception]
toIntegerMilliseconds[q_]:=Catch[Round[1000*toRealSeconds[q]],CURLLink`Utilities`Exception]

errorHandler[___]:=False
End[] (* End Private Context *)

EndPackage[]