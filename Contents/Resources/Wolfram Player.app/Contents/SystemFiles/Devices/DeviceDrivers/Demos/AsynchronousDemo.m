(* $Id$ *)

(* A demo for DeviceExecuteAsynchronous. Also implements properties. *)


BeginPackage["DeviceAPI`Drivers`Demos`AsynchronousDemo`"];

Begin["`Private`"];

(*----------- modified example from ref/URLFetchAsynchronous ----------*)

properties[_][_] = {};

eventFunction[devHandle_][_, "cookies", data_] := (
	properties[devHandle]["LastAccess"] = Date[];
	properties[devHandle]["Cookies"] = data
)

getProp[devHandle_,prop_] := properties[devHandle][prop]
setProp[devHandle_,prop_,rhs_] := properties[devHandle][prop] = rhs

fetch[handles:{_,devHandle_},url_,opts___?OptionQ] := 
	fetch[handles, url, eventFunction[devHandle], opts]

fetch[_,args__] := URLFetchAsynchronous[args]

async[_,"Read",url_,efun_] := URLFetchAsynchronous[url,efun]
async[_,"Save",url_,file_,opts___,efun_] := URLSaveAsynchronous[url,file,efun,opts]

		   
(*-----------------------------------------------------------------*)  

DeviceFramework`DeviceClassRegister["AsynchronousDemo",
	"ExecuteAsynchronousFunction" -> async,
	"NativeProperties" -> {"Cookies", "LastAccess"},
	"GetNativePropertyFunction" -> getProp,
	"SetNativePropertyFunction" -> setProp,
	"CloseFunction" -> (Quiet[
		properties[ #[[2]] ]["LastAccess"] =.;
		properties[ #[[2]] ]["Cookies"] =.;
	]&),
	"DriverVersion" -> 0.001
];

End[];

EndPackage[];
