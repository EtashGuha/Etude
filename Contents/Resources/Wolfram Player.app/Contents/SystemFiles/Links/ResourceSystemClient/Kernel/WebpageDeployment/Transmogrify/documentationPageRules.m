ResourceShingleTransmogrify`Private`documentationPageRules;

BeginPackage["ResourceShingleTransmogrify`"]
Begin["`Private`"]


$magnification = 2;
$defaultWidth = 725;
$maxAltTextSize = 70000;
$maxImageWidth = $defaultWidth;
$maxImageHeight = 500;
$maxDescriptionLength = 60;
$panelFont = "Source Sans Pro";


$stylesheet =
  Notebook @ {
      Cell[ StyleData[ StyleDefinitions -> "Default.nb" ] ],
      Cell[ StyleData[ "DialogStyle"        ], FontFamily -> $panelFont ],
      Cell[ StyleData[ "ManipulateLabel"    ], FontFamily -> $panelFont ],
      Cell[ StyleData[ "Manipulator"        ], FontFamily -> $panelFont ],
      Cell[ StyleData[ "Message"            ], FontFamily -> $panelFont ],
      Cell[ StyleData[ "MSG"                ], FontFamily -> $panelFont ],
      Cell[ StyleData[ "OSLText"            ], FontFamily -> $panelFont ],
      Cell[ StyleData[ "Panel"              ], FontFamily -> $panelFont, FontSize -> 10 ],
      Cell[ StyleData[ "PanelLabel"         ], FontFamily -> $panelFont ],
      Cell[ StyleData[ "RibbonText"         ], FontFamily -> $panelFont ],
      Cell[ StyleData[ "SuggestionsBarText" ], FontFamily -> $panelFont ],
      Cell[ StyleData[ "StandardForm"       ], FontFamily -> "Source Sans Pro" ]
  };


$rasterNotebookOpts := {
    StyleDefinitions -> $stylesheet,
    Magnification -> $magnification,
    WindowSize -> { $defaultWidth, $defaultWidth } * $magnification,
    "AutoStyleOptions" -> { "UndefinedSymbolStyle" -> { FontColor -> GrayLevel[ 0 ] } }
};


flatRecurse[ a___ ] := Flatten @ List @ Recurse @ a;


inputText::textfail = "Conversion to input text failed for `1`.";

inputText[ cell: Cell[ _, "Input", ___ ] ] /; ByteCount @ cell < $maxAltTextSize :=
  Module[ { inText, string },
      inText = FrontEndExecute @ ExportPacket[ cell, "InputText" ];
      string = Replace[ inText, { { str_String, ___ } :> str, ___ :> "" } ];
      StringReplace[ removeLineBreaks @ string, "\"" -> "&quot;" ]
  ];

inputText[ ___ ] :=
  "";


removeLineBreaks[ string_String ] :=
  StringReplace[ string,
                 RegularExpression @ "(?ms)\\s+(?:[\n\r])+\\s+" :> " "
  ];



rasterizeCell[ cell_Cell ] :=
  Rasterize @ Notebook[ { cell }, Sequence @@ $rasterNotebookOpts ];


cellToImg[ cell_Cell ] :=
    Module[ { img, wN, hN, wP, hP, pad, w, h, png, b64, src, text, attr },

        img = rasterizeCell @ cell;
        { wN, hN } = ImageDimensions @ img;
        wP = wN + Mod[ wN, $magnification ];
        hP = hN + Mod[ hN, $magnification ];
        pad = ImageCrop[ img, { wP, hP }, Padding -> GrayLevel[ 1, 0 ] ];
        w = ToString @ Ceiling[ wN / $magnification ];
        h = ToString @ Ceiling[ hN / $magnification ];
        png = ImportString[ ExportString[ pad, "PNG" ], "String" ];
        b64 = StringReplace[ ExportString[ png, "Base64" ], "\n" -> "" ];
        src = "data:image/png;base64," <> b64;
        text = inputText @ cell;

        attr = DeleteCases[ { "src"    -> src,
                              "alt"    -> text,
                              "width"  -> w,
                              "height" -> h,
                              "data-alt-length" -> StringLength @ text
                            },
                            _ -> ""
               ];

        XMLElement[ "img", attr, { } ]
  ];


cellToImg[ b_BoxData ] := cellToImg @ Cell @ b;
cellToImg[ s_String ] := cellToImg @ Cell @ s;

cellToImg[ box_Symbol[ args___ ] ] /; StringEndsQ[ SymbolName @ Unevaluated @ box, "Box" ] :=
  With[ { xml = Quiet @ Check[ cellToImg @ Cell[ BoxData @ box @ args, "Output" ], $Failed ] },
      xml /; MatchQ[ xml, XMLElement[ "img", { "src" -> _String, "alt" -> _String }, { } ] ]
  ];

cellToImg[ ___ ] := "";


ele[ tag_, None, data__ ] :=
  XMLElement[ tag, { }, Flatten @ { data } ];

ele[ tag_, class_, data__ ] :=
  XMLElement[ tag, { "class" -> class }, Flatten @ { data } ];

ele[ tag_, class_ ] :=
  ele[ tag, class, flatRecurse[ ] ];



img[ tag_, class_ ] :=
  ele[ tag, class, cellToImg @ SelectSelf[ ] ];



$functionRules := $functionRules =
  DeployedResourceShingle`Private`transmogrifyRules[ "Function" ];



fTransmogrify[ content_ ]:=
  ResourceShingleTransmogrify[ { content }, $functionRules ];



table[ ] :=
  Module[ { grid, cols, rows },

      grid = First @ SelectSelf[ ];

      cols = Map[ ele[ "td", None, fTransmogrify @ # ] &,
                  grid,
                  { 2 }
             ];

      rows = Map[ ele[ "tr", None, # ] &,
                  cols,
                  { 1 }
             ];

      ele[ "table", "grid", rows ]
  ];



inputCellImage[ cell_ ] :=
  ele[ "table", "example input",
       ele[ "tr", None, {
            ele[ "td", "in-out", "In[" <> getInputLabel @ cell <> "]:=" ],
            ele[ "td", None,
                 ele[ "div", "img-frame", cellToImg @ cell ]
            ]
       } ]
  ];


outputCellImage[ cell_ ] :=
  ele[ "table", "example output",
       ele[ "tr", None, {
            ele[ "td", "in-out", "Out[" <> getOutputLabel @ cell <> "]=" ],
            ele[ "td", None,
                 ele[ "div", "img-frame", cellToImg @ cell ]
            ]
       } ]
  ];


img[ "table", "example input" ] :=
  inputCellImage @ SelectSelf[ ];

img[ "table", "example output" ] :=
  outputCellImage @ SelectSelf[ ];



$refURL = "https://reference.wolfram.com/language/";


button[ "Hyperlink", target_String ] :=
  XMLElement[ "a", { "href" -> target }, flatRecurse[ ] ];

button[ "Hyperlink", _[ target_, ___ ] ] :=
  button[ "Hyperlink", target ];


button[ _, tgt_String? (StringStartsQ @ "paclet:ref/") ] :=
  Module[ { url },
      url = StringReplace[ tgt, StartOfString~~"paclet:" :> $refURL ] <> ".html";
      XMLElement[ "a", { "class" -> "reflink", "href" -> url }, flatRecurse[ ] ]
  ];

button[ "Link", url_String? (StringStartsQ @ "http") ] :=
  XMLElement[ "a", { "class" -> "reflink", "href" -> url }, flatRecurse[ ] ];

button[ "Link", target_String? (StringStartsQ[ "paclet:" ]) ] :=
  button[ "Link", $refURL <> StringTrim[ target, "paclet:"|".html" ] <> ".html" ];

button[ "Link", target_String ] :=
  button[ "Link", $refURL <> target ];

button[ "Link", None ] :=
  Module[ { content },
      content = First @ SelectChildren[ ];
      If[ StringQ @ content && NameQ[ "System`" <> content ]
          ,
          XMLElement[ "a",
                      {
                          "class" -> "reflink symbol",
                          "href" -> $refURL <> "ref/" <> content <> ".html"
                      },
                      flatRecurse[ ]
          ]
          ,
          flatRecurse[ ]
      ]
  ];

button[ ___ ] :=
  button[ "Hyperlink", "#" ];



documentationPageRules = XMLTransform @ {
    { Cell    , "FunctionIntro"          } :> ele[ "div" , "functionIntro"          ],
    { Cell    , "FunctionIntroWrap"      } :> ele[ "div" , "functionIntroWrap"      ],
    { Cell    , "Message"                } :> img[ "div" , "message"                ],
    { Cell    , "Print"                  } :> img[ "div" , "print"                  ],
    { Cell    , "Echo"                   } :> img[ "div" , "echo"                   ],
    { Cell    , "Notes"                  } :> ele[ "div" , "note"                   ],
    { Cell    , "UsageDescription"       } :> ele[ "p"   , "code-description"       ],
    { Cell    , "UsageInputs"            } :> ele[ "p"   , "code"                   ],
    { Cell    , "DetailsAndOptions"      } :> ele[ "div" , "details-and-options"    ],
    { Cell    , "Notes"                  } :> ele[ "div" , "notes"                  ],
    { Cell    , "TableNotes"             } :> ele[ "div" , "table-notes"            ],
    { Cell    , "TableText"              } :> ele[ "span", "table-text"             ],
    { Cell    , "Item"                   } :> ele[ "div" , "item"                   ],
    { Cell    , "Section"                } :> ele[ "h2"  , None                     ],
    { Cell    , "Subsection"             } :> ele[ "h3"  , None                     ],
    { Cell    , "Subsubsection"          } :> ele[ "h4"  , None                     ],
    { Cell    , "Subsubsubsection"       } :> ele[ "h5"  , None                     ],
    { Cell    , None                     } :> ele[ "span", "inline-cell"            ],
    { StyleBox, "TI"                     } :> ele[ "i"   , "ti"                     ],
    { StyleBox, "TR"                     } :> ele[ "i"   , "tr"                     ],
    { StyleBox, "ResourceFunctionHandle" } :> ele[ "span", "resourceFunctionHandle" ],
    { StyleBox, "CharRawComma"           } :> ele[ "span", "charRawComma"           ],
    { StyleBox, "CharEllipsis"           } :> ele[ "span", "charEllipsis"           ],
    { StyleBox, "CharRawLeftBracket"     } :> ele[ "span", "charRawLeftBracket"     ],
    { StyleBox, "CharRawRightBracket"    } :> ele[ "span", "charRawRightBracket"    ],
    { StyleBox, "CharRawLeftBrace"       } :> ele[ "span", "charRawLeftBrace"       ],
    { StyleBox, "CharRawRightBrace"      } :> ele[ "span", "charRawRightBrace"      ],
    { Cell    , "Input"                  } :> img[ "table", "example input"         ],
    { Cell    , "Output"                 } :> img[ "table", "example output"        ],
    { GridBox                            } :> table[ ]
    ,
    { GraphicsBox | PanelBox | PaneBox | Graphics3DBox | OverlayBox } :> cellToImg @ SelectSelf[ ],
    { InterpretationBox } :> cellToImg @ SelectSelf[ ]
    ,
    { ButtonBox } :> button[ GetStyle[ ], GetOption @ ButtonData ]
    ,
    { StyleBox } :>
      With[ { slant = GetOption @ FontSlant, weight = GetOption @ FontWeight },
          Which[ slant === Italic || slant === "Italic",
                 XMLElement[ "i", { }, flatRecurse[ ] ]
                 ,
                 weight === Bold || weight === "Bold",
                 XMLElement[ "b", { }, flatRecurse[ ] ]
                 ,
                 True,
                 XMLElement[ "span", { "class" -> "stylebox" }, flatRecurse[ ] ]
          ]
      ]
};

End[]
EndPackage[]

ResourceShingleTransmogrify`Private`documentationPageRules
