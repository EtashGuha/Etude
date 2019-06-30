
(* Based on Script code from HelpBrowserSearch *)

$currentResultSet = {};
currentPhrase = "";

(* Here we setup event bindings on widgets in the user interface
   onto functions defined in this script file *)

(* hitting return in text field will initiate searching *)
BindEvent[{"lookupTextField", "action"}, 
  Script[ performLookup[]; ],
  InvokeThread -> "New"];

(* combo boxes and check boxes will initiate searching *)
BindEvent[{"autoNavigationSelected", "action"},
  Script[ autoOpenSelectedItem[]; ],
  InvokeThread -> "New"];

BindEvent[{"wholeNameSelected", "action"},
  Script[ performLookup[]; ],
  InvokeThread -> "New"];

BindEvent[{"lookupCategoryNamesQ", "action"},
  Script[ performLookup[]; ],
  InvokeThread -> "New"];

BindEvent[{"lookupInCategory", "action"},
  Script[ performLookup[]; ],
  InvokeThread -> "New"];


(* start the search countdown whenever the lookup field changes*)
BindEvent[{"myDocument", "document"},
  Script[ InvokeMethod[{"myTimer", "start"}]; ],
  InvokeThread -> "New"
];


BindEvent[{"myTimer", "action"},
  Script[ Module[{currentStr},
    currentStr = PropertyValue[{"lookupTextField", "text"}];
    If[currentStr === previousStr,
      InvokeMethod[{"myTimer", "stop"}];
      performLookup["auto"],
      previousStr = currentStr
    ];
  ]],
  InvokeThread -> "New"
];


(* If the mouse event is a double click, open the selected row's notebook
   We name the resulting listener created from BindEvent because we need to remove it 
   as a listener during a search
*)
BindEvent[{"notebookResultTable", "mouseClicked"},
  Script[ 
    displaySelectedItemInfo[];
    If[ PropertyValue[{"#", "clickCount"}] >= 2,
       openSelectedItem[] ]; 
    ],
  Name -> "tableClickListener"];

(* TODO till we resolve the thread safety issue with "#" in a multi-threaded app, 
   ask for last ordering property from tableModel directly *)
BindEvent[{"notebookResultTableModel", "tableSorted"},
  Script[ 
    ord = PropertyValue[{"notebookResultTableModel", "ordering"}] + 1;
    $currentResultSet = $currentResultSet[[ord]];
    displaySelectedItemInfo[];
    ]
  ];
  
BindEvent[{"notebookResultTableSelectionModel", "valueChanged"},
  Script[
    displaySelectedItemInfo[];
    ]
  ];

BindEvent[{"notebookResultTable", "keyPressed"},
  Script[
    If[InvokeMethod[{"#", "getKeyText"}, PropertyValue[{"#", "keyCode"}]] === "Enter",
      openSelectedItem[];
      InvokeMethod[{"#", "consume"}]
    ]
  ]
];


(* These components are disabled while a search is in progress 
   since we are multithreaded and we do not want multiple kernel requests to happen *)
(*
searchDisabledWidgets = {
  "autoNavigationSelected",
  "wholeNameSelected",
  "lookupCategoryNamesQ",
  "lookupInCategory"
};
*)



performLookup[x___] := Module[{useCategories, multiWord, usePhrase, str, min},

   useCategories = PropertyValue[{"lookupInCategory", "selectedIndex"}];
   Which[
     useCategories === 0, useCategories = $HelpCategories,
     useCategories > 8, useCategories = $HelpCategories,
     True, useCategories = {Part[$HelpCategories, useCategories]}
   ];
   
   categoryQ = PropertyValue[{"lookupCategoryNamesQ", "selected"}];
   autoNavQ = PropertyValue[{"autoNavigationSelected", "selected"}];
   partialMatchQ = PropertyValue[{"wholeNameSelected", "selected"}] === False;
   
   If[{x}==={"auto"}, min = 3; min = 1, min = 1];
   
   usePhrase = PropertyValue[{"lookupTextField", "text"}];
   If[ StringLength[usePhrase] < min || currentSettings === 
       {usePhrase, categoryQ, partialMatchQ, useCategories},
     Return[]
   ];
   
   
   (* During search we do not want mouse clicks in the table to try calling into Mathematica *)
   InvokeMethod[{"notebookResultTable", "removeMouseListener"}, WidgetReference["tableClickListener"],
     InvokeThread -> "Dispatch"];
   
   SetPropertyValue[{"resultSummary", "text"}, "\"" <> usePhrase <> "\" ...",
     InvokeThread -> "Dispatch"];
   
   SetPropertyValue[{"notebookResultTable", "sortingEnabled"}, False, InvokeThread -> "Dispatch"];
        
   $currentResultSet = 
    CheckAbort[
      UseFrontEnd[
        ItemLookup[
          usePhrase,
          categoryQ,
          True,
          ItemLookupCategories -> useCategories,
          PartialMatch -> partialMatchQ
          ]
        ], 
      $Aborted];
      
   currentPhrase = usePhrase;
   currentSettings = {usePhrase, categoryQ, partialMatchQ, useCategories};
   
   (* Here we populate the table and status bar with our result data *)
   updateResults[];

   setResultSummary[ usePhrase, Length[$currentResultSet], categoryQ ];
   
   If[Length[$currentResultSet] === 0,
     SetPropertyValue[{"statusResultText","text"}, " "],
     displaySelectedItemInfo[];
   ];
   
   SetPropertyValue[{"notebookResultTable", "sortingEnabled"}, True, InvokeThread -> "Dispatch"]; 

   (* We can now allow mouse clicks in table *)
   InvokeMethod[{"notebookResultTable", "addMouseListener"}, WidgetReference["tableClickListener"],
     InvokeThread -> "Dispatch"];

];


setStatusResultText[str_String] :=
  SetPropertyValue[{"statusResultText", "text"}, str, InvokeThread -> "Dispatch"];


setResultSummary[str_String, n_Integer, catQ_] := 
Block[{s},
  s = StringForm["\"`1`\" found in `2` item`3`name`4`:",
    str, n, If[catQ, " and category ", " "], If[n===1,"","s"]];
  SetPropertyValue[{"resultSummary", "text"}, ToString[s], InvokeThread -> "Dispatch"]
];



displaySelectedItemInfo[] :=
Module[{row, lis, tag},
  row = PropertyValue[{"notebookResultTable", "selectedRow"}];
  If[ row < 0 || row+1 > Length[$currentResultSet], Return[]];
  
  {lis, tag} = $currentResultSet[[ row+1 ]];
  lis = Rest[lis];
  
  If[tag === Last @ lis, tag = "", tag = " (" <> tag <> ")"];
    
  setStatusResultText[ StringJoin[BoxForm`Intercalate[lis, " - "], tag] ];
  
  autoOpenSelectedItem[];
];



autoOpenSelectedItem[] :=
  If[ PropertyValue[{"autoNavigationSelected", "selected"}], openSelectedItem[] ];

openSelectedItem[] := 
Module[{row},
  row = PropertyValue[{"notebookResultTable", "selectedRow"}];
  If[ row < 0 || row+1 > Length[$currentResultSet], Return[]];
    
  openSelectedLocation[
    $currentResultSet[[row + 1, 1, 2]],
    $currentResultSet[[row + 1, 2]]
  ];
];

(* This goes to the beginning of an item location *)
openSelectedLocation[category_, indexTag_] :=
  FrontEndExecute[{
    FrontEnd`HelpBrowserLookup[ 
      AuthorTools`Experimental`Private`categoryToName[category], indexTag] }];
      

createResultRows[{{_,"Add-ons & Links",cat_,___,leaf_}, _}] := {leaf, "Add-ons & Links - " <> cat};
createResultRows[{{_,cat_,___, leaf_}, _}] := {leaf, cat};

createResultRows[{{_,"Add-ons & Links",cat_,___,leaf_}, _}] := {"Add-ons & Links - " <> cat, leaf};
createResultRows[{{_,cat_,___, leaf_}, _}] := {cat, leaf};


updateResults[] := (
    If[ Length[$currentResultSet] > 0,
      SetPropertyValue[{"notebookResultTableModel", "items"}, 
        createResultRows /@ $currentResultSet,
        InvokeThread -> "Dispatch"];
      ,
      SetPropertyValue[{"notebookResultTableModel", "rowCount"}, 0,
      InvokeThread -> "Dispatch"];
      ];
     
   InvokeMethod[{"notebookResultTable", "changeSelection"}, 0, 0, False, False,
     InvokeThread -> "Dispatch"];
   );

SetPropertyValue[{"notebookResultTableModel", "columnIdentifiers"}, {"Category", "Item"}];
SetPropertyValue[{"notebookResultTable", "autoCreateColumnsFromModel"}, False];
