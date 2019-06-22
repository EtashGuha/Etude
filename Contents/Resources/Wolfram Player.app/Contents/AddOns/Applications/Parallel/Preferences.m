(* :Title: Preferences.m -- persistent configuration of Parallel Tools *)

(* :Context: Parallel`Preferences` *)

(* :Author: Roman E. Maeder *)

(* :Summary:
   a framework for managing preferences for PT
   uses ResourceLocator for persistent storage
   This package can be read without reading all of PT
 *)

(* :Package Version: 1.0 alpha  *)

(* :Mathematica Version: 7 *)

(* :History:
   1.0 for PCT 3.0
*)

BeginPackage["Parallel`Preferences`"]

Preferences
InitializePreferences::usage = "InitializePreferences[] deletes an existing prefs file."
{add,get,set,clear,list,exists,load}
Scope
addPreference

prefs::usage = "prefs[method][...] implements the preference system methods."

tr::usage = "tr[\"key\"] looks up a localized string."

debugQ::usage = "debugQ[] returns True, if debug mode should be enabled."
debugPreference

Begin["`Private`"]

`$PackageVersion = 1.0;
`$thisFile = $InputFileName

Needs["ResourceLocator`"]

scopes = {session,user,default} = {"Session", "User", "Default"}; (* scopes *)
methods = {add,get,set,clear,list,exists,load}
actions = {adder, getter, setter, clearer,lister,checker,loader}
`handler (* key for storing handlers *)
`application (* key for storing app name *)
category = "Preferences" (* resource locator category for preferences *)
defaultScope = user
app = "Parallel"
`nopref (* signal missing pref *)

scopeQ[scope_] := MemberQ[scopes, scope]

(* localization support for palette, and status window *)

`$parallelRoot = DirectoryName[System`Private`$InputFileName]
tr = TextResourceLoad[ "Parallel", $parallelRoot]
Protect[tr]


(* method helpers *)

adder[obj_][name_->val_, hdl_:None] :=
Module[{},
	If[ hdl=!=None, obj[handler][name] = hdl ];
	setter[obj][name->val, Scope->default]; (* go through all the motions to init properly *)
	name -> getter[obj][name]
]

mergeStore[obj_, scope_][name_, val_] := obj[scope] = Append[DeleteCases[obj[scope], name->_], name->val]
clearStore[obj_, scope_][name_] := obj[scope] = DeleteCases[obj[scope], name->_]

getter[obj_][name_] := name /. Flatten[obj/@scopes] /. name:>Throw[nopref[name], Preferences]

checker[obj_][name_] := !MatchQ[Catch[getter[obj][name], Preferences], nopref[_]]

setter[obj_][name_->val_] := setter[obj][name->val, Scope->defaultScope]
setter[obj_][name_->val_, Scope->scope_?scopeQ] :=
Module[{newval},
	newval = obj[handler][name][name, val]; (* call handler *)
	If[ newval===$Failed || Head[newval]===obj[handler][name], Return[$Failed] ];
	mergeStore[obj, scope][name, newval];
	Scan[ clearStore[obj, #][name]&, TakeWhile[scopes, #=!=scope&] ];
	If[ scope =!= default, saveStore[obj]]; (* race condition, do not save during add, before reading user prefs *)
	newval
]

clearer[obj_][name_] := clearer[obj][name, Scope->defaultScope]
clearer[obj_][name_, Scope->scope_?scopeQ] :=
Module[{},
	Scan[ clearStore[obj, #][name]&, TakeWhile[scopes, #=!=scope&] ];
	If[ scope =!= default, clearStore[obj, scope][name] ]; (*not builtin ones *)
	saveStore[obj];
	name
]

lister[obj_][] := Union[ First /@ obj[default], First /@ obj[user] ]

(* always save at user scope for now *)
saveStore[obj_] :=
Block[{$Context="Parallel`Preferences`", $ContextPath={"System`"}}, (* careful about symbol scopes *)
	(* don't try to write for Player(Pro). An unset variable defaults to True *)
	If[ Parallel`Static`$persistentPrefs === True,
		PreferencesWrite[obj[application], category, obj[user]]
	];
]

(* init values from system and user scope *)
(* new symbols in Kernel/Preferences.m should end up in Parallel`Preferences` *)

`initdone=False;
loader[obj_][] := If[!initdone,
	With[{app=obj[application]},
		Block[{addPreference = obj[add], $Context="Parallel`Preferences`", $ContextPath={"System`"}}, (* symbols! *)
			Get[app<>"`Kernel`Preferences`"]; (* read built-in defaults *)
			If[ Parallel`Static`$persistentPrefs === True,
				(* only user scope for now *)
				obj[user] = PreferencesRead[app, category] /. {Null -> {}, $Failed -> {}};
			];
	]]; initdone=True;
]

(* hand-crafted factory method *)

With[{object=prefs},
	MapThread[(object[#1] = #2[object])&, {methods, actions}]; (* set up the methods *)
	(* init the values at all scopes *)
	(object[#]={})& /@ scopes;
	(* aux methods *)
	object[handler][_] :=  Function[{name, val}, val]; (* default handler does nothing *)
	object[application] = app; (* remember app name *)
	object
]

InitializePreferences[] := PreferencesDelete[ prefs[application], category ]

(* hack for debugging mode; duplicate the logic for choosing master kernel from Palette.m;
   used in Kernel/autoload *)

debugQ[] := Module[{master, name},
	prefs[load][];
	master = Which[
		ValueQ[Parallel`Static`$Profile] && (StringQ[Parallel`Static`$Profile] || Parallel`Static`$Profile===Automatic),
			Parallel`Static`$Profile,
		StringQ[Quiet[CurrentValue["RunningEvaluator"]]],
			CurrentValue["RunningEvaluator"] /. "Local" -> Automatic,
		True,
			"Batch"
	];
	name = debugPreference[master];
	(* effectively use a built-in default of True *)
	If[prefs[exists][name], TrueQ[prefs[get][name]], True]
]


End[]

EndPackage[]
