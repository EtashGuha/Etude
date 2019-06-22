(* Mathematica Package *)

(* Created by the Mathematica IDE May 17, 2005 *)



BeginPackage["ResourceLocator`", {"PacletManager`"}]
(* Exported symbols added here with SymbolName::usage *)

ResourcesLocate::usage = "ResourcesLocate[type] returns a list of locations of resources of type found in \
in Mathematica applications."

ResourceAdd::usage = "ResourceAdd[ dir, type] adds a resource of type. \
ResourceAdd[ dir, type, app] associates the resource with an application."

ApplicationDirectoryAdd::usage = "ApplicationDirectoryAdd[ dir] adds dir to the list \
of directories that are searched for Mathematica applications."

ApplicationDirectoriesLocate::usage = "ApplicationDirectoriesLocate[ ] returns a list of the applications directories."

ApplicationAdd::usage = "ApplicationAdd[dir] adds an application."
ApplicationsLocate::usage = "ApplicationsLocate[ ] returns a list of applications."

ApplicationDataDirectory::usage = "ApplicationDataDirectory[ appname, opts] returns an area for an application to store data."
ApplicationDataUserDirectory::usage = "ApplicationDataUserDirectory[ appname, opts] returns an area for an application to store data."

CreateDataDirectory::usage = "CreateDataDirectory is an option for ApplicationDataDirectory and ApplicationDataUserDirectory that states whether the data directory should be created."

PreferencesRead::usage = "PreferencesRead[ appname] returns the preference expression for the given application." <>
"PreferencesRead[ appname, categeory] returns the preferences for a given category."

PreferencesWrite::usage = "PreferencesWrite[ appname, expr] saves the preference expression for the given application." <>
"PreferencesWrite[ appname, categeory, expr] writes the preferences for a given category."

PreferencesDelete::usage = "PreferencesDelete[ appname] deletes the preference expression for the given application." <>
"PreferencesDelete[ appname, categeory delets the preferences for a given category."


TextResourceLoad::usage = "TextResourceLoad[ appName, appRoot] loads text resources for the application rooted at appRoot. It returns a TextResourceFunction."

TextResourceFunction::usage = "TextResourceFunction[ ] is a function that looks up text resources."

RegisterListener::usage = "RegisterListener[ type, listener] adds listener to be called when resources of type are added."


Begin["`Private`"]
(* Implementation of the package *)

If[!ValueQ[appDirectoryList], appDirectoryList = {}];
If[!ValueQ[appList], appList = {}];
resourceTable[_] = {};


getSubdirectories[ dir_] := Select[ FileNames[ "*", dir], DirectoryQ]


(*
 Add a directory in which to search for Applications.
 tgayley: I think this should be considered deprecated. PacletDirectoryAdd is the modern function.
*)
ApplicationDirectoryAdd[ dir_] :=
	Module[ {},
		If[ !MemberQ[ appDirectoryList, dir],
			PrependTo[ appDirectoryList, dir]];
		runListeners[ getSubDirectories[dir]];
		(* Until there are other hooks into the kernel, add app dirs to $Path
           so that Get["App`"] will find them. It is a hack to prepend them,
           done just so that a newer paclet downloaded will take precedence
           over an older one.
        *)
		If[ !MemberQ[ $Path, dir],
			PrependTo[ $Path, dir]];
		appDirectoryList
	]

(*
 Return the list of directory in which to search for Applications
*)
ApplicationDirectoriesLocate[] :=
	appDirectoryList

(*
 Add an application,  takes the path to the application
*)
ApplicationAdd[ dir_String] :=
	(
		If[ !MemberQ[ appList, dir],
			PrependTo[ appList, dir]];
		runListeners[ dir];
		(* Until there are other hooks into the kernel, add app dirs to $Path
           so that Get["App`"] will find them.
        *)
		If[!MemberQ[$Path, dir],
			PrependTo[$Path, ParentDirectory[dir]]
		]
	)

(*
 Return a list of all the applications known to the ResourceLocator.
 tgayley: This is currently not updated for paclet functionality. That is, it doesn't find
 applications that are paclets installed in the repository. What, exactly is an "application"
 paclet, anyway? Could be a paclet with a Kernel extension. But this whole ResourceLocator package isn't
 really about kernel applications at all, but apps with other types of resources like Documentation indices
 and DatabaseLink files. Frankly, I don't think this function is very important or well-defined. I doubt
 it gets any use outside the internals of this package, so I will ignore it for now.
*)
ApplicationsLocate[ appName_String:""] :=
	Module[ {appPaths, paths},
		appPaths = ApplicationDirectoriesLocate[];
		paths = Select[Flatten[FileNames["*", #]& /@ appPaths], DirectoryQ];
		paths = Join[ paths, appList];
		If[ appName === "", paths, Select[paths, Last[FileNameSplit[#]] == appName &]]
	]


(*
 Search through the applications known to the ResourceLocator to find resources 
 of this given type.  Also, search with the paclet manager. 
 tgayley: We want to give the PacletManager precedence, meaning that we only want the ResourceLocator's internal 
 ApplicationDirectories[]-based lookup to find resources belonging to apps that are unknown
 to the PacletManager. The PM does a lot of work to find the correct paclet based on version number, systemid, 
 etc., and we don't want a simple $Path-based system to include resources from inappropriate paclets. Thus, we
 let the PM resources come first, and delete ones found by the RL if they have the same app/paclet name as earlier ones. 
*)
ResourcesLocate[ type_String] :=
	Module[ {pacletResources, appResources, pacletNames},
		appResources = 
            {ExpandFileName[#1], #2} & @@@
			    Join[
					Select[{ToFileName[{#}, type], FileNameTake[#]}& /@ ApplicationsLocate[], DirectoryQ[First[#]]&],
					resourceTable[type]
				];
        (* This gets: {{"/path/to/res", "PacletName"}, {"/path/to/res", "PacletName"}, ...} where
           PacletName can be repeated (a paclet can provide more than one resource item). The Cases call just
           allows the result to be ignored if PacletManager has not been loaded (to support -nopaclet operation).
        *)
        pacletResources = Cases[PacletManager`Package`resourcesLocate[type], {_String, _String}];
		(* Merge these lists by dropping all resources found by the "app" method that match the paclet name of a
		   resource found by the paclet method.
		*)
		pacletNames = DeleteDuplicates[pacletResources[[All, -1]]];
		Join[pacletResources, Select[appResources, !MemberQ[pacletNames, Last[#]]&]]
	]


(*
  Search for a resource using the appName (might be a paclet name).
  This is not very efficient for paclets, as it finds all resources of the requested type in all paclets, then
  culls by name. Better to first find the specific paclet and then get just its resources. But it works in the current
  form, and I don't know if this function gets much, if any, use.
*)
ResourcesLocate[ {appName_String, type_}] :=
	Module[ {res},
		res = ResourcesLocate[type];
		Select[res, Last[#] == appName &]
	]

ResourceAdd[ dir_, type_, app_:None] :=
	resourceTable[ type] = Append[ resourceTable[type], {dir, app}];


(* Add these so their order reflects the traditional order in $Path. Ones
   added later come earlier in search path.
*)

ApplicationDirectoryAdd[
	ToFileName[{$InstallationDirectory, "SystemFiles"}, "Links"]]

ApplicationDirectoryAdd[
	ToFileName[{$InstallationDirectory, "AddOns"}, "Applications"]]

ApplicationDirectoryAdd[
	ToFileName[{$InstallationDirectory, "AddOns"}, "Autoload"]]

ApplicationDirectoryAdd[
	ToFileName[{$InstallationDirectory, "AddOns"}, "Packages"]]

(* These dirs not allowed in protected mode (i.e., the plugin). *)
If[!Developer`$ProtectedMode,
    ApplicationDirectoryAdd[
	   ToFileName[$BaseDirectory, "Applications"]];

    ApplicationDirectoryAdd[
	   ToFileName[$BaseDirectory, "Autoload"]];

    ApplicationDirectoryAdd[
	   ToFileName[$UserBaseDirectory, "Applications"]];

    ApplicationDirectoryAdd[
	   ToFileName[$UserBaseDirectory, "Autoload"]];
]

(*
 Listener functionality.   Handlers of resources can call 
 RegisterListener[ resourceType, listener] to get updates 
 on changes to directories that affect, the resource.
*)

RegisterListener[ type_String, listener_] :=
	Module[ {obj},
		obj = resourceListenerRegistry[ type];
		If[ obj === Null,
				obj = makeListener[ type];
				resourceListenerRegistry[ type] = obj];
		obj["addListener"][listener]
	]
	
	
runListeners[ dirs_List] := Scan[ runListeners, dirs]

runListeners[ dir_] :=
	Module[ {subdirs, dirTypes, listeners},
		subdirs = getSubdirectories[ dir];
		dirTypes = Map[ FileNameTake[#, -1]&, subdirs];
		listeners = Map[ resourceListenerRegistry, dirTypes];
		Scan[ #["callListeners"][ dir]&, listeners]
	]


resourceListenerRegistry[_] := Null




makeListener[type_] :=
	Module[ {obj},
		obj["type"] = type;
		obj["listeners"] = {};
		obj["addListener"][ fun_] := (obj["listeners"] = Append[ obj["listeners"], fun]);
		obj["removeListener"] [ fun_] := (obj["listeners"] = DeleteCases[ obj["listeners"], fun]);
		obj["callListeners"][ args___] := (Scan[ #[ obj, args]&, obj["listeners"]]);
		obj
	]



(*
 ApplicationDataUserDirectory and ApplicationDataDirectory
*)

General::appname = "The name `1` is not valid for the application. A valid name starts with a letter and is followed by letters and digits."

Options[ ApplicationDataUserDirectory] = {CreateDataDirectory -> True}

ApplicationDataUserDirectory[ name_String, OptionsPattern[]] :=
	applicationDirectoryImpl[ name,
		ApplicationDataUserDirectory,
		$UserBaseDirectory,
		OptionValue[CreateDataDirectory]]


applicationDirectoryImpl[ name_, head_, baseDir_, createQ_] :=
	Module[ {dir},
		If[ !testName[ name],
				Message[head::appname, name];
				Return[ $Failed]];
		dir = ToFileName[ {baseDir, "ApplicationData"}, name];
		If[ createQ && !DirectoryQ[dir], CreateDirectory[ dir, CreateIntermediateDirectories -> True]];
		dir
	]

testName[x_] := StringMatchQ[ x, LetterCharacter ~~ WordCharacter ..]



(*
  Preferences functionality
*)

General::file = "The preferences file `1` is not a file."
General::prefdir = "The preferences directory `1` cannot be created."
General::catname = "The name `1` is not valid for the category. A valid name starts with a letter and is followed by letters and digits."


PreferencesRead[ name_String, category_String:"Default"] :=
	Module[ {prefFile},
		prefFile = getPreferenceFile[ name, category, PreferencesRead];
		If[ prefFile === $Failed, Return[ $Failed]];
		If[ FileType[ prefFile] === None,
				{},
				Get[ prefFile]]
	]

PreferencesWrite[ name_String, category_String:"Default", expr_] :=
	Module[ {prefFile},
		prefFile = getPreferenceFile[ name, category, PreferencesWrite];
		If[ prefFile === $Failed, Return[ $Failed]];
		Write[ prefFile, expr];
		Close[ prefFile];
	]

PreferencesDelete[ name_String, category_String:"Default"] :=
	Module[ {prefFile},
		prefFile = getPreferenceFile[ name, category, PreferencesDelete];
		If[ prefFile === $Failed, Return[ $Failed]];
		If[ FileType[ prefFile] === File, DeleteFile[ prefFile]];
	]



getPreferenceFile[ name_String, category_String, head_] :=
	Module[ {prefDir, prefFile},
		If[ !testName[ name],
				Message[head::appname, name];
				Return[ $Failed]];
		If[ !testName[ category],
				Message[head::catname, category];
				Return[ $Failed]];
		dir = ApplicationDataUserDirectory[ name];
		prefDir =  ToFileName[ {dir}, "Preferences"];
		Switch[ FileType[ prefDir],
					File,
						Message[ head::prefDir, prefDir];
						Return[ $Failed],
					None,
						CreateDirectory[ prefDir]];
		prefFile = ToFileName[ {prefDir}, category <> ".m"];
		If[ DirectoryQ[ prefFile],
				Message[ head::file, prefFile];
				$Failed,
				prefFile]
	]



(*
  TextResource Functionality
*)

TextResourceLoad::nores = "Application `1` cannot find text resource file `2`.";
TextResourceLoad::badres = "Data `1` for application `2` in text resource file `3` are not a list of rules.";
TextResourceLoad::badres1 = "Data `1` for application `2` in text resource file `3` for language `4` are not a list of rules.";


TextResourceLoad[ appName_String, appRoot_String] :=
	Module[{resName, resBase, resFile, dataLang, data, def},
		resName = "Default.m";
		resBase = ToFileName[ {appRoot}, "TextResources"];
		resFile = ToFileName[ {resBase, $Language}, resName];
		If[ $Language === "English" || FileType[ resFile] =!= File, 
				dataLang = {},	
				dataLang = Get[ resFile]];
		If[ !ListQ[ dataLang],
			Message[ TextResourceLoad::badresl, dataLang, appName, resName, $Language];
			Return[ $Failed]];
		resFile = ToFileName[ {resBase, "English"}, resName];
		If[ FileType[ resFile] =!= File,
			Message[ TextResourceLoad::nores, appName, resName];
			Return[ $Failed]];
		data = Get[ resFile];
		If[ !ListQ[ data],
			Message[ TextResourceLoad::badres, data, appName, resName];
			Return[ $Failed]];
		def = Verbatim[ _] /. dataLang;
		If[ !StringQ[ def], 
			def = Verbatim[ _] /. data];
		If[ !StringQ[ def], def = "Not Found"];
		dataLang = DeleteCases[ dataLang, _[Verbatim[_], _]];
		data = DeleteCases[ data, _[Verbatim[_], _]];
		TextResourceFunction[ dataLang, data, def]
	]


TextResourceFunction[dataLang_, data_, def_][ key_String] := 
	Module[ {val},
		val = key /. dataLang;
		If[ val === key,
			val = key /. data];
		If[ val === key,
			val = def];
		val
	]




Format[ TextResourceFunction[ dataLang_, data_, def_]] := Row[ {"<<", TextResourceFunction, ">>"}]

End[]

EndPackage[]

