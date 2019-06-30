BeginPackage["GraphStore`IRI`", {"GraphStore`"}];

AbsoluteIRIQ;

CompactIRI;

ExpandIRI;

FileToIRI;
Options[FileToIRI] = {
	OperatingSystem :> $OperatingSystem
};

IRIQ;

Begin["`Private`"];

AbsoluteIRIQ[i_] := StringMatchQ[
	i /. (IRI | URL)[s_] :> s,
	StartOfString ~~ Alternatives[
		(* absolute-IRI   = scheme ":" ihier-part [ "?" iquery ] *)
		LetterCharacter ~~ (LetterCharacter | DigitCharacter | "+" | "-" | ".") ... ~~ ":" ~~ Except[" "] ...,
		(* ipath-absolute = "/" [ isegment-nz *( "/" isegment ) ] *)
		"/" ~~ Except["/" | " "] ~~ Except[" "] ...
	]
];
CompactIRI[args___] := With[{res = Catch[iCompactIRI[args], $failTag]}, res /; res =!= $failTag];
ExpandIRI[args___] := With[{res = Catch[iExpandIRI[args], $failTag]}, res /; res =!= $failTag];
FileToIRI[args___] := With[{res = Catch[iFileToIRI[args], $failTag]}, res /; res =!= $failTag];
IRI /: Import[IRI[iri_?StringQ], rest___] := Import[iri, rest];
IRI /: MakeBoxes[IRI[iri_String?StringQ], fmt_] := RowBox[{"IRI", "[", TemplateBox[{MakeBoxes[iri, fmt]}, "URLArgument"], "]"}];
IRIQ[i_String] := StringQ[URLParse[i, "Scheme"]] && StringFreeQ[i, " "];
IRIQ[IRI[i_String]] := IRIQ[i];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


(* -------------------------------------------------- *)
(* compact IRI *)

clear[iCompactIRI];
iCompactIRI[i_String, None] := i;
iCompactIRI[i_String, base_String] := (
	If[UnsameQ @@ URLParse[{i, base}, {"Scheme", "User", "Domain", "Port"}],
		Return[i]
	];
	Module[
		{iPath, iQuery, iFragment, bPath, result},
		{iPath, iQuery, iFragment} = URLParse[i, {"Path", "QueryString", "Fragment"}];
		If[iQuery === None && StringContainsQ[i, "?"],
			iQuery = "";
		];
		If[iFragment === None && StringEndsQ[i, "#"],
			iFragment = "";
		];
		bPath = URLParse[base, "Path"];
		While[iPath =!= {} && bPath =!= {} && First[iPath] === First[bPath] && (! MatchQ[bPath, {Except[""]}] || StringQ[iQuery] || StringQ[iFragment]),
			iPath = Rest[iPath];
			bPath = Rest[bPath];
		];
		If[bPath =!= {},
			iPath = Join[ConstantArray["..", Length[bPath] - 1], iPath];
		];
		If[iPath === {""},
			PrependTo[iPath, "."];
		];
		result = StringJoin[Riffle[iPath, "/"]];
		If[StringQ[iQuery],
			result = result <> "?" <> iQuery;
		];
		If[StringQ[iFragment],
			result = result <> "#" <> iFragment;
		];
		result
	]
);
iCompactIRI[x___, IRI[i_String], y___] := iCompactIRI[x, i, y];
iCompactIRI[x___, f_File, y___] := iCompactIRI[x, iFileToIRI[f], y];

(* end compact IRI *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* expand IRI *)

(* Function naming: ExpandIRI versus AbsoluteIRI? *)
(* ExpandFileName does not check whether a file exists. *)
(* AbsoluteFileName checks whether a file with the resulting name exists. *)

(* Return value: "Always string" versus "always IRI[string]" versus "depending on the first argument"? *)

(* Uniform Resource Identifier (URI): Generic Syntax *)
(* https://tools.ietf.org/html/rfc3986 *)


(* 3.2.  Authority *)
clear[composeAuthority];
composeAuthority[{user_, host_, port_}] := Module[
	{result},
	result = "";
	If[StringQ[user],
		result = result <> user <> "@";
	];
	result = result <> host;
	If[StringQ[port],
		result = result <> ":" <> port;
	];
	result
];


(* 5.2.  Relative Resolution *)
clear[iExpandIRI];
iExpandIRI[i_String, None] := i;
iExpandIRI[r_String, base_String] := Module[
	{rScheme, rAuthority, rPath, rQuery, rFragment, tScheme, tAuthority, tPath, tQuery, tFragment},
	(* 5.2.2.  Transform References *)
	{rScheme, rPath, rQuery, rFragment} = URLParse[r, {"Scheme", "PathString", "QueryString", "Fragment"}];
	(* restore empty query and fragment *)
	(* https://bugs.wolfram.com/show?number=349084 *)
	If[rQuery === None && StringContainsQ[r, "?"],
		rQuery = "";
	];
	If[rFragment === None && StringEndsQ[r, "#"],
		rFragment = "";
	];
	rAuthority = URLParse[r, {"User", "Domain", "Port"}];
	(* reverse case normalization *)
	(* https://bugs.wolfram.com/show?number=349083 *)
	If[StringQ[rAuthority[[2]]],
		StringReplace[r, x : rAuthority[[2]] :> (rAuthority[[2]] = x;), 1, IgnoreCase -> True]
	];
	If[StringQ[rScheme],
		tScheme = rScheme;
		tAuthority = rAuthority;
		tPath = removeDotSegments[URLParse[r, "PathString"]];
		tQuery = rQuery;
		,
		If[StringQ[rAuthority[[2]] (* "Domain" *)],
			tAuthority = rAuthority;
			tPath = removeDotSegments[URLParse[r, "PathString"]];
			tQuery = rQuery;
			,
			If[rPath === "",
				tPath = URLParse[base, "PathString"];
				If[StringQ[rQuery],
					tQuery = rQuery;
					,
					tQuery = URLParse[base, "QueryString"];
				];
				,
				If[StringStartsQ[rPath, "/"],
					tPath = removeDotSegments[rPath];
					,
					tPath = merge[URLParse[base, "PathString"], rPath];
					tPath = removeDotSegments[tPath];
				];
				tQuery = rQuery;
			];
			tAuthority = URLParse[base, {"User", "Domain", "Port"}];
		];
		tScheme = URLParse[base, "Scheme"];
	];
	tFragment = rFragment;
	(* always include a (potentially empty) authority component in file URIs *)
	If[tScheme === "file" && ! StringQ[tAuthority[[2]]],
		tAuthority[[2]] = "";
	];
	composeComponents[tScheme, tAuthority, tPath, tQuery, tFragment]
];
iExpandIRI[x___, IRI[i_String], y___] := iExpandIRI[x, i, y];
iExpandIRI[x___, f_File, y___] := iExpandIRI[x, iFileToIRI[f], y];

(* 5.2.3.  Merge Paths *)
clear[merge];
merge[basePath_String, rPath_String] := If[basePath === "",
	"/" <> rPath,
	If[StringEndsQ[basePath, "/"],
		basePath <> rPath,
		StringReplace[basePath, "/" ~~ Except["/"] .. ~~ EndOfString :> "/"] <> rPath
	]
];

(* 5.2.4.  Remove Dot Segments *)
clear[removeDotSegments];
removeDotSegments[path_String] := Module[
	{in, out},
	If[StringFreeQ[path, "."],
		Return[path];
	];
	(* 1 *)
	in = path;
	out = "";
	(* 2 *)
	While[in =!= "",
		Which[
			(* A *)
			StringStartsQ[in, "../" | "./"],
			in = StringDelete[in, StartOfString ~~ "../" | "./"],
			(* B *)
			StringStartsQ[in, "/./" | ("/." ~~ EndOfString)],
			in = StringReplace[in, StartOfString ~~ "/./" | "/." -> "/", 1],
			(* C *)
			StringStartsQ[in, "/../" | ("/.." ~~ EndOfString)],
			in = StringReplace[in, "/../" | "/.." -> "/", 1];
			out = StringReplace[out, {
				StartOfString ~~ x___ ~~ "/" ~~ Shortest[___] ~~ EndOfString :> x,
				___ :> ""
			}, 1],
			(* D *)
			in === "." || in === "..",
			in = "",
			(* E *)
			True,
			StringReplace[
				in,
				StartOfString ~~ s : Repeated["/", {0, 1}] ~~ x : Except["/"] ... ~~ rest___ ~~ EndOfString :> (
					out = out <> s <> x;
					in = rest;
				)
			];
		];
	];
	(* 3 *)
	out
];


(* 5.3.  Component Recomposition *)
clear[composeComponents];
(* https://bugs.wolfram.com/show?number=349074 *)
(*composeComponents[scheme_, authority_List, path_String, query_, fragment_] := URLBuild[<|
	"Scheme" -> scheme,
	"User" -> authority[[1]],
	"Domain" -> authority[[2]],
	"Port" -> authority[[3]],
	"PathString" -> path,
	"QueryString" -> query,
	"Fragment" -> fragment
|>];*)
composeComponents[scheme_, authority_List, path_String, query_, fragment_] := Module[
	{result},
	result = "";
	If[StringQ[scheme],
		result = result <> scheme <> ":";
	];
	If[StringQ[authority[[2]] (* "Domain" *)],
		result = result <> "//" <> composeAuthority[authority];
	];
	result = result <> path;
	If[StringQ[query],
		result = result <> "?" <> query;
	];
	If[StringQ[fragment],
		result = result <> "#" <> fragment;
	];
	result
];

(* end expand IRI *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* file to IRI *)

(* https://tools.ietf.org/html/rfc8089 *)
clear[iFileToIRI];
Options[iFileToIRI] = Options[FileToIRI];
iFileToIRI[file_String, OptionsPattern[]] := Module[
	{res},
	res = file;
	If[OptionValue[OperatingSystem] === $OperatingSystem,
		res = ExpandFileName[res];
	];
	res = FileNameSplit[res, OperatingSystem -> OptionValue[OperatingSystem]];
	If[res =!= {} && OptionValue[OperatingSystem] === "Windows",
		res[[1]] = If[StringStartsQ[First[res], "\\\\"],
			(* Windows network file *)
			StringReplace[StringDrop[First[res], 2], "\\" -> "/"],
			(* Windows local file *)
			"/" <> First[res]
		];
	];
	res = FileNameJoin[res, OperatingSystem -> "Unix"];
	res = StringReplace[res, " " -> "%20"];
	res = "file://" <> res;
	res
];
iFileToIRI[File[file_String], rest___] := iFileToIRI[file, rest];

(* end file to IRI *)
(* -------------------------------------------------- *)


End[];
EndPackage[];
