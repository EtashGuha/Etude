BeginPackage["Compile`ResourceLoader`"]


SaveResource
LoadResource


Begin["`Private`"]


	
LoadResource::found = "Value `1` of the CompileResourceDirectory option is not set to an existing directory."
SaveResource::create = "The setting for the CompileResourceDirectory option `1` cannot be created."
LoadResource::nres = "The resource `1` cannot be found."


(*
  TODO issue message
*)
getResourcesDir[ head_, dir_] :=
	Null

getResourcesDir[ head_, Automatic] :=
	FileNameJoin[ {Compile`Utilities`$CompileRootDirectory, "CompileResources"}]

getResourcesDir[ head_, dir_String] :=
	dir





Options[SaveResource] = {"CompileResourceDirectory" -> Automatic}

(*
  Note the setting of $ContextPath,  this makes sure that 
  fully qualified symbols are written.
*)
SaveResource[name_, data_, opts:OptionsPattern[]] :=
	Module[ {dir, resFile},
		dir = getResourcesDir[ SaveResource, OptionValue["CompileResourceDirectory"]];
		If[ dir === Null,
			Return[]];
		If[ FileType[dir] =!= Directory,
			CreateDirectory[dir]];
		If[ FileType[dir] =!= Directory,
			Message[ SaveResource::create, dir];
			Return[]];
		resFile = FileNameJoin[ {dir, name <> ".wl"}];
		Block[{$ContextPath = {"System`"}},
			Put[ data, resFile]];
		resFile
	]


Options[LoadResource] = {"CompileResourceDirectory" -> Automatic}

LoadResource[name_, opts:OptionsPattern[]] :=
	Module[ {dir, resFile},
		dir = getResourcesDir[ LoadResource, OptionValue["CompileResourceDirectory"]];
		If[ dir === Null,
			Return[]];
		If[ FileType[dir] =!= Directory,
			Message[ LoadResource::found, dir];
			Return[]];
		resFile = FileNameJoin[ {dir, name <> ".wl"}];
		Quiet[Get[resFile]]
	]



End[]

EndPackage[] 

