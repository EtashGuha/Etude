(* :Title: NETBlock.m *)

(* :Author:
        Todd Gayley
        tgayley@wolfram.com
*)

(* :Package Version: 1.7 *)

(* :Mathematica Version: 5.0 *)
             
(* :Copyright: .NET/Link source code (c) 2003-2019, Wolfram Research, Inc. All rights reserved.

   Use is governed by the terms of the .NET/Link license agreement, which can be found at
   www.wolfram.com/solutions/mathlink/netlink.
*)

(* :Discussion:
    
   This file is a component of the .NET/Link Mathematica source code.
   It is not a public API, and should never be loaded directly by users or programmers.

   .NET/Link uses a special system wherein one package context (NETLink`) has its implementation
   split among a number of .m files. Each component file has its own private context, and also
   potentially introduces public symbols (in the NETLink` context) and so-called "package" symbols,
   where the term "package" comes from Java terminology, referring to symbols that are visible
   everywhere within the implementation of .NET/Link, but not to clients. The NETLink.m file itself
   is produced by an automated tool from the component files and contains only declarations.
   
   Do not modify the special comment markers that delimit Public- and Package-level exports.
*)


(*<!--Public From NETBlock.m

NETBlock::usage =
"NETBlock[expr] causes all new .NET objects returned to the Wolfram Language during the evaluation of expr to be released when expr \
finishes. It is an error to refer to such an object after NETBlock ends. See the usage message for ReleaseNETObject for more \
information. NETBlock only affects new objects, not additional references to ones that have previously been seen. NETBlock is \
a way to mark a set of objects as temporary so they can be automatically cleaned up on both the Wolfram Language and .NET sides."

BeginNETBlock::usage =
"BeginNETBlock[] and EndNETBlock[] are equivalent to the NETBlock function, except that they work across a larger span than \
the evaluation of a single expr. Every BeginNETBlock[] must have a paired EndNETBlock[]."

EndNETBlock::usage =
"BeginNETBlock[] and EndNETBlock[] are equivalent to the NETBlock function, except that they work across a larger span than \
the evaluation of a single expr. Every BeginNETBlock[] must have a paired EndNETBlock[]."

ReleaseNETObject::usage =
"ReleaseNETObject[netobject] tells the .NET memory-management system to forget any references to the specified NETObject \
that are being maintained solely for the sake of the Wolfram Language. The NETObject in the Wolfram Language is no longer valid after the call. \
You call ReleaseNETObject when you are completely finished with an object in the Wolfram Language, and you want to allow it to be \
garbage-collected in .NET."

KeepNETObject::usage =
"KeepNETObject[object] causes the specified object(s) not to be released when the current NETBlock ends. \
KeepNETObject allows an object to \"escape\" from the current NETBlock. It only has an effect if the object was in fact \
slated to be released by that block. The object is promoted to the \"release\" list of the next-enclosing NETBlock, \
if there is one. The object will be released when that block ends (unless you call KeepNETObject again in the outer block). \
KeepNETObject[object, Manual] causes the specified object to escape from all enclosing NETBlocks, meaning that the object \
will only be released if you manually call ReleaseNETObject."

LoadedNETObjects::usage =
"LoadedNETObjects[] returns a list of all the .NET objects that have been loaded into the current session."

LoadedNETTypes::usage =
"LoadedNETTypes[] returns a list of all the .NET types that have been loaded into the current session."

LoadedNETAssemblies::usage =
"LoadedNETAssemblies[] returns a list of all the .NET assemblies that have been loaded into the current session."

-->*)

(*<!--Package From NETBlock.m

addToNETBlock
resetNETBlock
findAliases

-->*)


(* Current context will be NETLink`. *)

Begin["`NETBlock`Private`"]


(*********************************  NETBlock/BeginNETBlock/EndNETBlock  ***********************************)

(* Note that it is safe to miss a call to EndNETBlock[] (for example, if user aborts out of NETBlock).
   The only consequence is that the NET objects in the block are not released. Note also that if I am
   willing to lose BeginNETBlock[] and EndNETBlock[] (which I probably am) then I can probably rewrite
   NETBlock in a simpler, safer way using Block to localize $netBlockRecord.
*)

If[!ValueQ[$netBlockRecord], $netBlockRecord = {}]

Internal`SetValueNoTrack[$netBlockRecord, True]

SetAttributes[NETBlock, HoldAllComplete]

NETBlock[e_, opts___?OptionQ] :=
	Module[{res},
		Internal`WithLocalSettings[BeginNETBlock[], res = e, EndNETBlock[res, opts]]
	]


BeginNETBlock[] := ($netBlockRecord = {$netBlockRecord};)

EndNETBlock[opts___?OptionQ] := EndNETBlock[Null, opts]

EndNETBlock[result_, opts___?OptionQ] :=
    Module[{release, keep, keptObjectsSlatedForRelease},
        If[$netBlockRecord =!= {},
            (* Second (=last) element of $netBlockRecord, if it exists, is a nested list of objects that were created
               in this NETBlock: {{{obj}, obj}, obj}
            *)
            {$netBlockRecord, release} = {First[$netBlockRecord], Flatten[Rest[$netBlockRecord]]};
            If[result =!= Null && NETObjectQ[result] && MemberQ[release, result],
                release = DeleteCases[release, result];
                (* Promote escaping object to the "release" list of next-higher NETBlock. *)
                addToNETBlock[result]
            ];
            ReleaseNETObject[release]
        ];
    ]

addToNETBlock[obj_] :=
    If[$netBlockRecord =!= {},
        If[Length[$netBlockRecord] == 1,
            (* First new object in this NETBlock. *)
            AppendTo[$netBlockRecord, {obj}],
        (* else *)
            (* Avoid appending to a growing list by instead adding objects by nesting {{old}, new} *)
            $netBlockRecord = {First[$netBlockRecord], {Last[$netBlockRecord], obj}}
        ]
    ]

resetNETBlock[] := $netBlockRecord = {}


(****************************************  ReleaseNETObject  ******************************************)

ReleaseNETObject[syms__] := ReleaseNETObject[{syms}]

ReleaseNETObject[syms_List] := 
    Module[{nsyms, aliasedSyms},
        nsyms = DeleteCases[Select[syms, NETObjectQ], Null];
        If[nsyms =!= {},
            runListeners[syms, "Release"];
            (* This gets rid of the reference that lives in the ObjectHandler.instanceCollection on the .NET side. *)
            nReleaseObject[nsyms];
            (* Now delete definitions for object symbols. For each object in nsyms that has an aliased version
               in existence, we search out all aliases and add them to the list of symbols for which definitions
               will be cleared. The hasAliasedVersions function is set via upvalues on the object symbols.
               It is an optimization to avoid checking for cast aliases for every symbol.
            *)
            aliasedSyms = Select[nsyms, hasAliasedVersions];
            If[aliasedSyms =!= {},
                nsyms = Union[Join[nsyms, Flatten[findAliases /@ aliasedSyms]]]
            ];
            Unprotect @@ nsyms;
            ClearAll @@ nsyms;
            Remove @@ nsyms
        ]
    ]

(* An object with a name like NETObject$1234$5678 is a casted version of another object named exactly NETObject$5678.
   The $1234 is a hash value of the name of the type to which it is cast to.
   This function finds all the objects NETObject$xxxx$yyyy or NETObject$yyyy given an object NETObject$yyyy
   or NETObject$xxxx$yyyy. In other words, it finds all objects with the same key value yyyy.
   If it turns out to be too slow to look up this information for each cast object, it could be recorded in
   createInstanceDefs, for example as castAliases[key_String] := {...}.
*) 
findAliases[sym_Symbol] :=
    Module[{name = SymbolName[sym], key},
        key = StringDrop[name, Last[Flatten[StringPosition[name, "$"]]]];
        (* Find all symbol names that end in the same key value. *)
        Symbol /@ Names["NETLink`Objects`NETObject*$" <> key]
    ]


(****************************************  KeepNETObject  ******************************************)

KeepNETObject::obj = "At least one argument to KeepNETObject was not a valid .NET object."

(* This form wouldn't be called by the user, but it might be called inside other definitions. *)
KeepNETObject[{}] = Null
KeepNETObject[{}, Automatic | Manual] = Null

KeepNETObject[objs__Symbol, man:(Automatic | Manual)] := KeepNETObject[{objs}, man]
KeepNETObject[objs__Symbol] := KeepNETObject[{objs}]

KeepNETObject[objs:{__?NETObjectQ}, man:(Automatic | Manual):Automatic] :=
    Module[{prevBlockRecord, release, objectsToKeep, aliasedObjects, keptObjectsSlatedForRelease},
        If[$netBlockRecord =!= {},
            (* We take the specified objects that were actually planned to be released in this NETBlock,
               remove them from the "release" list of the current NETBlock, and add them to the release list
               of the parent NETBlock. If Manual, however, simply remove the objects from anywhere
               in $netBlockRecord.
            *)
			objectsToKeep = objs;
            (* Because objects can appear in multiple aliases (NETObject$yyyy, NETObject$xxxx$yyyy, etc.), for
               any object that has such aliases, we must remove all its aliases from the release list.
               Recall that only a single refcount entry exists for an object no matter how many aliases it
               exists in.
            *)
            aliasedObjects = Select[objectsToKeep, hasAliasedVersions];
			Scan[
				(objectsToKeep = objectsToKeep ~Join~ findAliases[#])&,
				aliasedObjects
			];
			objectsToKeep = Union[objectsToKeep];
			If[man === Manual,
				(* Completely remove objects from $netBlockRecord. They will never be freed via the
				   NETBlock mechanism, only by manual call to ReleaseNETObject.
				*)
				$netBlockRecord = DeleteCases[$netBlockRecord, Alternatives @@ objectsToKeep, Infinity],
			(* else *)
                {prevBlockRecord, release} = {First[$netBlockRecord], Flatten[Rest[$netBlockRecord]]};
                (* The list objectsToKeep now contains the original set of object symbols requested, plus any
                   uncasted versions of objects that were casted.
                *)
                keptObjectsSlatedForRelease = Cases[release, Alternatives @@ objectsToKeep];
                release = Complement[release, keptObjectsSlatedForRelease];
                Which[
                    prevBlockRecord === {},
                        (* There is no outer NETBlock to promote to. The objects escape for good. *)
                        $netBlockRecord = {{}, release},
                    prevBlockRecord === {{}},
                        (* Outer NETBlock has had no objects introduced into its release list yet. *)
                        $netBlockRecord = {{{}, keptObjectsSlatedForRelease}, release},
                    True,
                        (* Outer NETBlock has a non-empty release list. prevBlockRecord looks like {{...}, {...}}. 
                           We insert the newly-promoted objects into the 2nd part of prevBlockRecord. It does not matter
                           that these keptObjectsSlatedForRelease are grouped in a list.
                        *)
                        $netBlockRecord = {Insert[prevBlockRecord, keptObjectsSlatedForRelease, {-1, -1}], release}
                ]
            ]
        ];
    ]
    
KeepNETObject[___] := Message[KeepNETObject::obj]


(****************************************  LoadedNETObjects  ******************************************)

LoadedNETObjects[] := (InstallNET[]; nPeekObjects[])


End[]
