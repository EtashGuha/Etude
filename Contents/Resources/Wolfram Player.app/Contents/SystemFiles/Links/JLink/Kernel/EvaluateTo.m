(* :Title: EvaluateTo.m *)

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
   The M code "EvaluateTo" functions called by the J/Link KernelLink API methods of the same names.
   They are public so that J/Link or other MathLink programmers can use them in their own programs.
	
   This file is a component of the J/Link Mathematica source code.
   It is not a public API, and should never be loaded directly by users or programmers.

   J/Link uses a special system wherein one package context (JLink`) has its implementation
   split among a number of .m files. Each component file has its own private context, and also
   potentially introduces public symbols (in the JLink` context) and so-called "package" symbols,
   where the term "package" comes from Java terminology, referring to symbols that are visible
   everywhere within the implementation of J/Link, but not to clients.
*)


(* No usage messages for these is deliberate. These are not called from an interactive session, only from
   Java code in a Java front end.
*)
EvaluateToImage
EvaluateToTypeset
$DefaultImageFormat
ConvertToCellExpression


Begin["`Package`"]
(* No Package-level exports, but Begin/End are needed by tools. *)
End[]


(* Current context will be JLink`. *)

Begin["`EvaluateTo`Private`"]


If[!ValueQ[$DefaultImageFormat], $DefaultImageFormat = "GIF"]


SetAttributes[EvaluateToImage, HoldFirst]

EvaluateToImage[e_, useFE:(True | False):False, opts___?OptionQ] :=
	EvaluateToImage[e, useFE, $DefaultImageFormat, opts]

EvaluateToImage[e_, useFE:(True | False):False, fmt_String:$DefaultImageFormat, opts___?OptionQ] :=
	Block[{$DisplayFunction = Identity, expr, format, ps, result},
		expr = If[StringQ[Unevaluated[e]], ToExpression[e], e];
		(* Support passing "Automatic" as format. For speed, it uses GIF for 2-D graphics,
		   where color limitations are not likely to be relevant, but JPEG for 3D images.
		   This is only relevant when not using the FE, as GIF produces fine results with the FE.
		*)
		format =
			If[fmt == "Automatic",
				Which[
					$VersionNumber >= 5.1,
						"GIF",  (* GIF is fastest in 5.1 and later; no quality loss. *)
					!useFE && (Head[e] === Graphics3D || Head[e] === SurfaceGraphics),
						"JPEG",
					True,
						"GIF"
				],
			(* else *)
				fmt
			];
		If[useFE && (FrontEndSharedQ[$ParentLink] || ConnectToFrontEnd[]),
			UseFrontEnd[
				Which[
					$VersionNumber >= 5.1,
						result = ExportString[expr, format, opts],
					format == "GIF" || format == "Metafile",
						ps = DisplayString[expr, "MPS", opts];
						If[StringQ[ps],
							LinkWrite[First[$FrontEnd], ExportPacket[Cell[GraphicsData["PostScript", ps], "Graphics"], format]];
							result = First[LinkRead[First[$FrontEnd]]]
						],
					True,
						result = ExportString[expr, format, opts]
				]
			],
		(* else *)
			If[format == "GIF" && $VersionNumber < 5.1,
				(* For GIF, DisplayString is faster than ExportString (because it uses
				   psrender). Other relevant formats (e.g., JPEG) are either equivalent
				   for these two functions or they are not even supported by DisplayString.
				*) 
				result = DisplayString[expr, format, opts],
			(* else *)
				result = ExportString[expr, format, opts]
			]
		];
		(* It's OK to pass back whatever garbage DisplayString returned if it wasn't a string,
		   but since it could be a huge expr, we avoid the MathLink overhead. *)
		If[StringQ[result], result, Null]
	]

SetAttributes[EvaluateToTypeset, HoldFirst]

EvaluateToTypeset[e_, frm_Symbol:StandardForm, pageWidth_Integer:0, opts___?OptionQ] :=
	EvaluateToTypeset[e, frm, pageWidth, $DefaultImageFormat, opts]

EvaluateToTypeset[e_, frm_Symbol:StandardForm, pageWidth_Integer:0, format_String, opts___?OptionQ] :=
	Block[{cellExpr, result},
		cellExpr = ConvertToCellExpression[e, frm, pageWidth];
		If[FrontEndSharedQ[$ParentLink] || ConnectToFrontEnd[],
			UseFrontEnd[
				If[format == "GIF" || format == "Automatic",
					LinkWrite[First[$FrontEnd], ExportPacket[cellExpr, "GIF"]];
					result = First[LinkRead[First[$FrontEnd]]],
				(* else *)
					result = ExportString[cellExpr, format, opts]
				]
			]
		];
		(* It's OK to pass back whatever garbage ExportString returned if it wasn't a string,
		   but since it could be a huge expr, we avoid the MathLink overhead. *)
		If[StringQ[result], result, Null]
	]


(* It is useful to split out the ConvertToCellExpression functionality from EvaluateToTypeset so it can be called separately. *)

SetAttributes[ConvertToCellExpression, HoldFirst]

ConvertToCellExpression[e_, frm_Symbol, pageWidth_Integer, cellOpts___?OptionQ] :=
	Block[{$DisplayFunction = Identity, expr, pWidth},
		expr = If[StringQ[Unevaluated[e]], ToExpression[e], e];
		pWidth = If[pageWidth > 0, pageWidth, Infinity];
		(* Expr will typically not be a Cell or BoxData, but we allow advanced users to send in exprs of these types
		   if they want to take more control over the process. If you supply a full Cell expr, it is passed unaltered
		   into Display (thus, you can set FontSize or any of the myriad Cell options yourself). If you supply a
		   BoxData expr, you get control over the ToBoxes step, and I add the Cell wrapper with its various options.
		*)
		Switch[expr,
			_Cell,
				expr,
			_BoxData,
				Cell[expr, "Output", ShowCellBracket->False, CellMargins->{{0,0},{0,0}}, PageWidth->pWidth, cellOpts],
			_,
				Cell[BoxData[ToBoxes[expr, frm]], "Output", ShowCellBracket->False, CellMargins->{{0,0},{0,0}}, PageWidth->pWidth, cellOpts]
		]
	]


End[]
