(* :Title: MathKernel.m *)

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


(*******************

    These are support functions for the MathKernel class. The computeWrapper function is called from .NET.
    
********************)

(*<!--Public From MathKernel.m

-->*)

(*<!--Package From MathKernel.m

computeWrapper

-->*)


(* Current context will be NETLink`. *)

Begin["`MathKernel`Private`"]

SetAtributes[computeWrapper, {HoldFirst}]


computeWrapper[input_, outputFormat_String, pageWidth_, imgFormat_String,
                    imgWidth_Integer, imgHeight_Integer, dpi_Integer, useFE:(True | False), captureGraphics:(True | False)] :=
    Block[{$DisplayFunction, expr},
        $DisplayFunction = If[captureGraphics, captureGraphicsDisplayFunction[imgFormat, imgWidth, imgHeight, dpi, useFE], Identity];
        expr = If[StringQ[Unevaluated[input]], ToExpression[input], input];
        Switch[outputFormat,
            "InputForm",
                ToString[expr, FormatType -> InputForm],
            "OutputForm",
                ToString[expr, FormatType -> OutputForm, PageWidth -> If[pageWidth > 0, pageWidth, Infinity]],
            "Expr",
                expr,
            "StandardForm" | "TraditionalForm",
                EvaluateToTypeset[expr, ToExpression[outputFormat], If[pageWidth > 0, pagewidth, 0], imgFormat,
                                    ImageResolution -> If[dpi > 0, dpi, Automatic],
                                    ImageSize -> {If[imgWidth > 0, imgWidth, Automatic], If[imgHeight > 0, imgHeight, Automatic]}]
        ]
    ]


captureGraphicsDisplayFunction[imgFormat_String, imgWidth_Integer, imgHeight_Integer, dpi_Integer, useFE:(True | False)] :=  
    Function[g,
        opts = {ImageResolution -> If[dpi > 0, dpi, Automatic],
                ImageSize -> {If[imgWidth > 0, imgWidth, Automatic], If[imgHeight > 0, imgHeight, Automatic]}
               };
        LinkWrite[$ParentLink, DisplayEndPacket[EvaluateToImage[g, useFE, imgFormat, Sequence @@ opts]]];
        g
    ]
    

End[]
