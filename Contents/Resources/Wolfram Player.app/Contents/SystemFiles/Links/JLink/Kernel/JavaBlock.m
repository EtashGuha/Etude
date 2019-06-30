(* :Title: JavaBlock.m *)

(* :Author:
        Todd Gayley
        tgayley@wolfram.com
*)

(* :Package Version: 4.9 *)

(* :Mathematica Version: 4.0 *)

(* :Copyright: J/Link source code (c) 1999-2019, Wolfram Research, Inc. All rights reserved.

   Use is governed by the terms of the J/Link license agreement, which can be found at
   www.wolfram.com/solutions/mathlink/jlink.
*)

(* :Discussion:
   JavaBlock functionality.

   This file is a component of the J/Link Mathematica source code.
   It is not a public API, and should never be loaded directly by users or programmers.

   J/Link uses a special system wherein one package context (JLink`) has its implementation
   split among a number of .m files. Each component file has its own private context, and also
   potentially introduces public symbols (in the JLink` context) and so-called "package" symbols,
   where the term "package" comes from Java terminology, referring to symbols that are visible
   everywhere within the implementation of J/Link, but not to clients.
*)



JavaBlock::usage =
"JavaBlock[expr] causes all new Java objects returned to the Wolfram Language during the evaluation of expr to be released when expr finishes. It is an error to refer to such an object after JavaBlock ends. See the usage message for ReleaseJavaObject for more information. JavaBlock only affects new objects, not additional references to ones that have previously been seen. If a JavaBlock returns a single JavaObject as a result, that object will not be released. JavaBlock is a way to mark a set of objects as temporary so they can be automatically cleaned up when the block of code ends."

BeginJavaBlock::usage =
"BeginJavaBlock[] and EndJavaBlock[] are equivalent to the JavaBlock function, except that they work across a larger span than the evaluation of a single expression. Every BeginJavaBlock[] must have a paired EndJavaBlock[]."

EndJavaBlock::usage =
"BeginJavaBlock[] and EndJavaBlock[] are equivalent to the JavaBlock function, except that they work across a larger span than the evaluation of a single expression. Every BeginJavaBlock[] must have a paired EndJavaBlock[]."

KeepObjects::usage =
"KeepObjects is a deprecated option to JavaBlock. Use the KeepJavaObject function instead."

KeepJavaObject::usage =
"KeepJavaObject[object] causes the specified object(s) not to be released when the current JavaBlock ends. KeepJavaObject allows an object to \"escape\" from the current JavaBlock, and it only has an effect if the object was in fact slated to be released by the current block. The object is promoted to the \"release\" list of the next-enclosing JavaBlock, if there is one, so it will be released when that block ends (unless you call KeepJavaObject again in the outer block). KeepJavaObject[object, Manual] causes the specified object to escape from all enclosing JavaBlocks, meaning that the object will only be released if you manually call ReleaseJavaObject."

ReleaseObject::usage =
"ReleaseObject is deprecated. The new name is ReleaseJavaObject."

ReleaseJavaObject::usage =
"ReleaseJavaObject[javaobject] tells the Java memory-management system to forget about any references to the specified JavaObject that are being maintained solely for the sake of the Wolfram Language. The JavaObject in the Wolfram Language is no longer valid after the call. You call ReleaseJavaObject when you are completely finished with an object in the Wolfram Language, and you want to allow it to be garbage-collected in Java."


Begin["`Package`"]

addToJavaBlock

End[]


(* Current context will be JLink`. *)

Begin["`JavaBlock`Private`"]


Options[JavaBlock] = Options[EndJavaBlock] = {KeepObjects -> {}}

(* Note that it is safe to miss a call to EndJavaBlock[] (for example, if user aborts out of JavaBlock).
   The only consequence is that the Java objects in the block are not released. Note also that if I am
   willing to lose BeginJavaBlock[] and EndJavaBlock[] (which I probably am) then I can probably rewrite
   JavaBlock in a simpler, safer way using Block to localize $javaBlockRecord.
*)

If[!ValueQ[$javaBlockRecord], $javaBlockRecord = {}]

Internal`SetValueNoTrack[$javaBlockRecord, True]


SetAttributes[JavaBlock, HoldAllComplete]

JavaBlock[e_, opts___?OptionQ] :=
	Module[{res},
		Internal`WithLocalSettings[BeginJavaBlock[], res = e, EndJavaBlock[res, opts]]
	]

BeginJavaBlock[] := ($javaBlockRecord = {$javaBlockRecord};)

EndJavaBlock[opts___?OptionQ] := EndJavaBlock[Null, opts]

EndJavaBlock[result_, opts___?OptionQ] :=
	Module[{release, keep, keptObjectsSlatedForRelease},
		If[$javaBlockRecord =!= {},
			JAssert[MatchQ[$javaBlockRecord, {_List} | {_List, _List}]];
			(* Second (=last) element of $javaBlockRecord, if it exists, is a nested list of objects that were created
			   in this JavaBlock: {{{obj}, obj}, obj}. It could also be {} if KeepJavaObject had been used on
			   all the objects created in the block.
			*)
			{$javaBlockRecord, release} = {First[$javaBlockRecord], Flatten[Rest[$javaBlockRecord]]};
			If[result =!= Null && JavaObjectQ[result] && MemberQ[release, result],
				release = DeleteCases[release, result];
				(* Promote escaping object to the "release" list of next-higher JavaBlock. *)
				addToJavaBlock[result]
			];
			(* Note that we use the options for JavaBlock. We'll probably deprecate EndJavaBlock, and most users
			   end up here by calling JavaBlock. Calling SetOptions for either JavaBlock or EndJavaBlock
			   is a very unlikely thing for a user to want to do anyway.
			*)
			keep = KeepObjects /. Flatten[{opts}] /. Options[JavaBlock];
			If[keep =!= {},
				keptObjectsSlatedForRelease = Cases[release, Alternatives @@ Flatten[{keep}]];
				release = Complement[release, keptObjectsSlatedForRelease];
				(* Promote escaping objects to the "release" list of next-higher JavaBlock. *)
				addToJavaBlock /@ keptObjectsSlatedForRelease
			];
			ReleaseJavaObject[release]
		];
	]

addToJavaBlock[obj_] :=
	If[$javaBlockRecord =!= {},
		If[Length[$javaBlockRecord] == 1,
			(* First new object in this JavaBlock. *)
			AppendTo[$javaBlockRecord, {obj}],
		(* else *)
			(* Avoid appending to a growing list by instead adding objects by nesting {{old}, new} *)
			$javaBlockRecord = {First[$javaBlockRecord], {Last[$javaBlockRecord], obj}}
		]
	]



ReleaseObject = ReleaseJavaObject  (* ReleaseObject is deprecated. *)

ReleaseJavaObject[syms__] := ReleaseJavaObject[{syms}]

ReleaseJavaObject[syms_List] :=
	Module[{nsyms, jvmSyms},
		nsyms = DeleteCases[Select[syms, JavaObjectQ], Null];
		(* Split up the objects according to what JVM they are from. In most cases, all the objects
		   to be released will be from one JVM, so don't bother to sort before calling SPlit.
		*)
		jvmSyms = Split[nsyms, jvmFromInstance[#1] == jvmFromInstance[#2] &];
		jReleaseObject[jvmFromInstance[First[#]], #]& /@ jvmSyms;
		If[nsyms =!= {},
			ClearAll @@ nsyms;
			Remove @@ nsyms
		]
	]


KeepJavaObject::obj = "At least one argument to KeepJavaObject was not a valid Java object."

(* This form wouldn't be called by the user, but it might be called inside other definitions. *)
KeepJavaObject[{}] = Null
KeepJavaObject[{}, Automatic | Manual] = Null

KeepJavaObject[objs__Symbol, man:(Automatic | Manual)] := KeepJavaObject[{objs}, man]
KeepJavaObject[objs__Symbol] := KeepJavaObject[{objs}]

KeepJavaObject[objs:{__?JavaObjectQ}, man:(Automatic | Manual):Automatic] :=
    Module[{prevBlockRecord, release, keptObjectsSlatedForRelease},
		Which[
			$javaBlockRecord === {},
				(* Nothing to do; no objects slated for release ever. *)
				Null,
			man === Manual,
				(* Completely remove objects from $javaBlockRecord. They will never be freed via the
				   JavaBlock mechanism, only by manual call to ReleaseObject.
				*)
				$javaBlockRecord = DeleteCases[$javaBlockRecord, Alternatives @@ objs, Infinity],
			True,
			    (* We take the specified objects that were actually planned to be released in this JavaBlock,
			       remove them from the "release" list of the current JavaBlock, and add them to the release list
			       of the parent JavaBlock.
			    *)
				{prevBlockRecord, release} = {First[$javaBlockRecord], Flatten[Rest[$javaBlockRecord]]};
				keptObjectsSlatedForRelease = Cases[release, Alternatives @@ objs];
				release = Complement[release, keptObjectsSlatedForRelease];
				Which[
				    prevBlockRecord === {},
				        (* There is no outer JavaBlock to promote to. The objects escape for good. *)
				        $javaBlockRecord = {{}, release},
				    Length[prevBlockRecord] == 1,
				        (* Outer JavaBlock has had no objects introduced into its release list yet. *)
				        $javaBlockRecord = {{{}, keptObjectsSlatedForRelease}, release},
				    True,
				        (* Outer JavaBlock has a non-empty release list. prevBlockRecord looks like {{...}, {...}}.
				           We insert the newly-promoted objects into the 2nd part of prevBlockRecord. It does not matter
				           that these keptObjectsSlatedForRelease are grouped in a list.
				        *)
				        $javaBlockRecord = {Insert[prevBlockRecord, keptObjectsSlatedForRelease, {-1, -1}], release}
				]
		];
	]

KeepJavaObject[___] := Message[KeepJavaObject::obj]


End[]
