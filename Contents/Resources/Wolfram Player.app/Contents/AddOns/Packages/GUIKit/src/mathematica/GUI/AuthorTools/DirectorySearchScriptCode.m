
(* Script code for DirectorySearch Widget

Features to consider:
 - column sorting by clicks
 - tree mode instead of table
 - custom column model for Count column to right justify number
 - column width resizing to pack default widths better
 - persistent preferences for widths, order, ascending/descending
 
*)
 
currentResultSet = {{}};
currentKeywords = {};

(* Here we setup event bindings on widgets in the user interface
   onto functions defined in this script file *)

(* hitting return in text field will initiate searching *)
BindEvent[{"searchTextField", "action"}, 
  Script[ performSearch[]; ],
  InvokeThread -> "New"];
  
(* search button click will initiate searching *)
BindEvent[{"searchButton", "action"}, 
  Script[ performSearch[]; ],
  InvokeThread -> "New",
  Name -> "searchActionListener"];
  
(* remove the cancel listener by default, search and cancel listeners are swapped in and out *)
InvokeMethod[{"searchButton", "removeActionListener"}, WidgetReference["cancelActionListener"]];

(* If the mouse event is a double click, open the selected row's notebook
   We name the resulting listener created from BindEvent because we need to remove it 
   as a listener during a search
*)
BindEvent[{"notebookResultTable", "mouseClicked"},
  Script[ 
    If[ PropertyValue[{"#", "clickCount"}] >= 2,
       openSelectedNotebook[] ]; 
    ],
  Name -> "tableClickListener"];
           
(* TODO till we resolve the thread safety issue with "#" in a multi-threaded app, 
   ask for last ordering property from tableModel directly *)
BindEvent[{"notebookResultTableModel", "tableSorted"},
  Script[ 
    ord = PropertyValue[{"notebookResultTableModel", "ordering"}] + 1;
    currentResultSet = Join[{First[currentResultSet][[ord]]}, Rest[currentResultSet]];
    ]
  ];
  
(* this toggles the current result mode *)
BindEvent[{"resultModeSelect", "action"}, 
  Script[ changeResultMode[]; ]
  ];


(* choose a directory with Experimental`FileBrowse *)
BindEvent[{"browseButton", "action"},
  Script[ changeTargetDirectory[]; ]
  ];



notebookSearchFormat = "Output";
cellStyleNamePrefix = "cellStyle_";

cellStyleNames = Select[ "WidgetNames" /. GUIInformation[], StringMatchQ[#, cellStyleNamePrefix <> "*"]&];

(* These components are disabled while a search is in progress 
   since we are multithreaded and we do not want multiple kernel requests to happen *)
searchDisabledWidgets = Join[
  {"resultModeSelect", "multiWordSelect", "recursionModeSelect",
  "caseSensitive", "option_WordSearch", "browseButton", "chosenDirectory"}, 
  cellStyleNames
  ];

performSearch[] := Module[{origTitle, useCellStyles, useDirectory, multiWord, useKeywords, str,
  ignoreCase, wordSearch, fileList},

   useCellStyles = StringDrop[#, StringLength[cellStyleNamePrefix]]& /@ 
    Select[ cellStyleNames, PropertyValue[{#, "selected"}]&];
   
   useDirectory = PropertyValue[{"chosenDirectory", "text"}];
   recurse = PropertyValue[{"recursionModeSelect", "selected"}];
   ignoreCase = Not @ TrueQ @ PropertyValue[{"caseSensitive", "selected"}];
   wordSearch = TrueQ @ PropertyValue[{"option_WordSearch", "selected"}];
   
   Which[
     FileType[useDirectory] === None,
     setStatusResultText["No search made. No such directory."]; Return[]
     ,
     FileType[useDirectory] === File,
     fileList = {useDirectory}
     ,
     fileList = FileNames["*.nb", useDirectory, If[recurse,Infinity,1]];
     fileList === {},
     setStatusResultText["No search made. Directory contains no notebooks."]; Return[]
     ,
     True,
     Null
   ];
   
   useKeywords = PropertyValue[{"searchTextField", "text"}];
   If[ StringLength[useKeywords] < 1, 
     setStatusResultText["No search made. You must enter a set of keywords to search by."];
     Return[];
     ];
   
   (* During search we do not want mouse clicks in the table to try calling into Mathematica *)
   InvokeMethod[{"notebookResultTable", "removeMouseListener"}, WidgetReference["tableClickListener"],
     InvokeThread -> "Dispatch"];
   
   origTitle = PropertyValue[{"frame", "title"}];
   SetPropertyValue[{"frame", "title"}, "Searching...",
     InvokeThread -> "Dispatch"];
   SetPropertyValue[{"searchTextField", "editable"}, False,
     InvokeThread -> "Dispatch"];
     
   Scan[ SetPropertyValue[{#, "enabled"}, False,
     InvokeThread -> "Dispatch"]&, searchDisabledWidgets];
   SetPropertyValue[{"notebookResultTable", "sortingEnabled"}, False, InvokeThread -> "Dispatch"];
   
   If[ MemberQ[useCellStyles, "All"], useCellStyles = All];

   useKeywords = PropertyValue[{"searchTextField", "text"}];
   str = StringToStream[useKeywords];
   useKeywords = ReadList[str, Word];

   multiWord =  PropertyValue[{"multiWordSelect", "selectedIndex"}];
   If[ multiWord === 0, multiWord = Or, multiWord = And];

   Off[NotebookFileOutline::nocache];
   
   (* Swap search/cancel state on button *)
   InvokeMethod[{"searchButton", "removeActionListener"}, WidgetReference["searchActionListener"],
     InvokeThread -> "Dispatch"];
   (* This is done to prevent button from changing size when text is changed *)
   SetPropertyValue[{"searchButton", "preferredSize"}, PropertyValue[{"searchButton", "size"}]];
   SetPropertyValue[{"searchButton", "text"}, "Cancel",
     InvokeThread -> "Dispatch"];
   InvokeMethod[{"searchButton", "addActionListener"}, WidgetReference["cancelActionListener"],
     InvokeThread -> "Dispatch"];
   
   (* Here is where we call AuthorTool's NotebookSearch after pulling in
      the function parameters values from the GUI dialog *)
  Block[{AuthorTools`Experimental`Private`beginProgress, AuthorTools`Experimental`Private`endProgress,
    AuthorTools`Experimental`Private`progressUpdateFunction},
    
    (* We don't want the notebook GUI from the default NotebookSearch *)
   AuthorTools`Experimental`Private`beginProgress[s___] := Null;
   AuthorTools`Experimental`Private`endProgress[] := Null;
   (* We do want to update our status text with the NotebookSearch progress callback function though *)
   AuthorTools`Experimental`Private`progressUpdateFunction[str_String] := setStatusResultText[str];
     
   currentResultSet = 
    CheckAbort[
      UseFrontEnd[
        NotebookSearch[ 
          fileList, 
          useKeywords, 
          notebookSearchFormat,
          MultiWordSearch -> multiWord,
          SelectedCellStyles -> useCellStyles,
          IgnoreCase -> ignoreCase,
          WordSearch -> wordSearch,
          EncodeSearchStrings -> False,
          Verbosity -> 5]
        ], 
      $Aborted];
    ];
      
   currentKeywords = useKeywords;
   
   (* Swap search/cancel state on button *)
   InvokeMethod[{"searchButton", "removeActionListener"}, WidgetReference["cancelActionListener"],
     InvokeThread -> "Dispatch"];
   SetPropertyValue[{"searchButton", "text"}, "Search",
     InvokeThread -> "Dispatch"];
   InvokeMethod[{"searchButton", "addActionListener"}, WidgetReference["searchActionListener"],
     InvokeThread -> "Dispatch"];
     
   If[ currentResultSet === $Aborted,
      currentResultSet = {{}, $Aborted};
      currentKeywords = {};
      ];
    
   On[NotebookFileOutline::nocache];

   (* Here we populate the table and status bar with our result data *)
   updateResults[];

   setStatusResultText[ createStatusString[Rest[currentResultSet]] ];
   
   SetPropertyValue[{"frame", "title"}, origTitle,
     InvokeThread -> "Dispatch"];
   SetPropertyValue[{"searchTextField", "editable"}, True,
     InvokeThread -> "Dispatch"];
     
   Scan[ SetPropertyValue[{#, "enabled"}, True,
     InvokeThread -> "Dispatch"]&, searchDisabledWidgets];
   SetPropertyValue[{"notebookResultTable", "sortingEnabled"}, True, InvokeThread -> "Dispatch"];

   (* We can now allow mouse clicks in table *)
   InvokeMethod[{"notebookResultTable", "addMouseListener"}, WidgetReference["tableClickListener"],
     InvokeThread -> "Dispatch"];

   InvokeMethod[{"searchTextField", "selectAll"},
     InvokeThread -> "Dispatch"];
   ];

setStatusResultText[str_String] :=
  SetPropertyValue[{"statusResultText", "text"}, str,
     InvokeThread -> "Dispatch"];
     
createStatusString[{t_, c_, n_, files_}] := ToString @ 
  Which[
    c===0, StringForm["No cells found in `1` notebooks.   `2` seconds", files, First[t]],
    c===1, StringForm["Found 1 cell in 1 of `1` notebooks.   `2` seconds", files, First[t]],
    n===1, StringForm["Found `1` cells in 1 of `2` notebooks.   `3` seconds", c, files, First[t]],
    True, StringForm["Found `1` cells in `2` of `3` notebooks.   `4` seconds", c, n, files, First[t]]
    ];
createStatusString[{$Aborted}] := "Search cancelled";
createStatusString[___] := "";

changeResultMode[] := Module[{},
  Switch[ PropertyValue[{"resultModeSelect", "selectedIndex"}],
    1,  (* Notebook *)
      mode = "Notebook";
      notebookSearchFormat = "Expressions";
      If[ TrueQ[InvokeMethod[{"frame", "isAncestorOf"}, WidgetReference["notebookResultScrollPane"]]],
        (* First time we attempt to remove the scrollpane we store its parent for future use *)
        If[ WidgetReference["resultParent"] === Null,
          PropertyValue[{"notebookResultScrollPane", "parent"}, Name -> "resultParent"] ];
        If[ WidgetReference["resultParent"] =!= Null,
          InvokeMethod[{"resultParent", "remove"}, WidgetReference["notebookResultScrollPane"]];
          changeNotebookFrameHeight[ - PropertyValue[{"notebookResultScrollPane", "height"}]];
          ];
        ];
    ,
    _,  (* Table *)
      mode = "Table";
      notebookSearchFormat = "Output";
      If[ !TrueQ[InvokeMethod[{"frame", "isAncestorOf"}, WidgetReference["notebookResultScrollPane"]]],
        If[ WidgetReference["resultParent"] =!= Null,
          InvokeMethod[{"resultParent", "remove"}, WidgetReference["statusResultText"]];
          InvokeMethod[{"resultParent", "add"}, WidgetReference["notebookResultScrollPane"]];
          InvokeMethod[{"resultParent", "add"}, WidgetReference["statusResultText"]];
          changeNotebookFrameHeight[ PropertyValue[{"notebookResultScrollPane", "height"}]];
          ];
        ];
      ];
  ];
  


changeTargetDirectory[] := Module[{dir},
  dir = Experimental`FileBrowse[False];
  If[dir === $Failed, Return[]];
  dir = DirectoryName[dir];
  SetPropertyValue[{"chosenDirectory", "text"}, dir]
];



changeNotebookFrameHeight[adjustment_] :=
  Module[{newWidth,newHeight},
  InvokeMethod[{"notebookResultScrollPane", "invalidate"}];
  If[ WidgetReference["resultParent"] =!= Null,
    InvokeMethod[{"resultParent", "invalidate"}];
    ];
  InvokeMethod[{"frame", "invalidate"}];
  newWidth = PropertyValue[{"frame", "width"}];
  newHeight = PropertyValue[{"frame", "height"}] + adjustment;
  SetPropertyValue[{"frame", "size"}, 
    Widget["Dimension", {"width" ->  newWidth, "height" -> newHeight}] ];
  InvokeMethod[{"frame", "validate"}];
  ];
  
openSelectedNotebook[] := 
  Module[{row, resultList = First[currentResultSet], dir, nbFile, opts, i, nbObj},
  row = PropertyValue[{"notebookResultTable", "selectedRow"}];
  If[ row < 0 || row+1 > Length[resultList], Return[]];
  
  nbFile = resultList[[row+1, 1]];
  opts = Rest[ resultList[[row+1]] ];
  i = First["Index" /. opts];
  dir = DirectoryName[nbFile];
  
  nbObj=NotebookOpen[nbFile];
  FrontEndExecute[{
    FrontEnd`SelectionMove[nbObj, Before, Notebook],
    FrontEnd`SelectionMove[nbObj, Next, Cell, i],
    FrontEnd`FrontEndToken[nbObj, "OpenSelectionParents"]
  }];
];


createResultRows[{nb_, opts__}] := Module[{cnt, shortName, dirName},
   dirName = DirectoryName[nb];
   shortName = StringReplace[nb, dirName -> ""];
   cnt = "Count" /. {opts};
   {shortName, dirName, cnt}
   ];

updateResults[] := Module[{model, resultList = First[currentResultSet]},
    
    If[ StringMatchQ[notebookSearchFormat, "Output"] &&
       Head[resultList] === List,
      If[ Length[resultList] > 0,
        SetPropertyValue[{"notebookResultTableModel", "items"}, 
                  createResultRows /@ resultList,
          InvokeThread -> "Dispatch"];
       ,
       SetPropertyValue[{"notebookResultTableModel", "rowCount"}, 0,
        InvokeThread -> "Dispatch"];
        ];
      ,
      SetPropertyValue[{"notebookResultTableModel", "rowCount"}, 0,
        InvokeThread -> "Dispatch"];
      ];
     
   InvokeMethod[{"notebookResultTable", "changeSelection"}, 0, 0, False, False,
     InvokeThread -> "Dispatch"];
   ];

SetPropertyValue[{"notebookResultTableModel", "columnIdentifiers"}, {"Notebook", "Directory", "Count"}];
SetPropertyValue[{"notebookResultTable", "autoCreateColumnsFromModel"}, False];
 