
$wizardResult = Null;

$wizardSource = Null;

BindEvent[{"wizard", "wizardFinished"}, 
  Script[
    $wizardResult = PropertyValue[{"tablePreviewModel", "items"}]; 
    If[ $wizardResult =!= Null && !TrueQ[ PropertyValue[{"ScriptEvaluator", "runningModal"}]],
      (* We choose to create a new notebook of the data if the wizard
         is running in a non-modal session otherwise the modal return result
         will be the value of $wizardResult *)
      NotebookPut[
       Notebook[{
         Cell[TextData[{"Text Import Wizard results:"}], "Text"], 
         Cell[BoxData[ToBoxes[$wizardResult]], "Input"]
         }]
       ];
      
      ];
    ] 
  ];
  
BindEvent[{"wizard", "wizardCanceled"}, 
  Script[ $wizardResult = Null;] 
  ];
  
BindEvent[{"wizard", "wizardDidReset"}, 
  Script[ $wizardResult = Null;] 
  ];
  
BindEvent["endModal",
  Script[ $wizardResult ]
  ];

BrowseFilename[] := Module[{returnValue},
  If[ WidgetReference["openFileDialog"] === Null,
    Widget["FileDialog", Name -> "openFileDialog"];
    SetPropertyValue[{"openFileDialog", "dialogTitle"}, "Select Source Text File"];
    SetPropertyValue[{"openFileDialog", "multiSelectionEnabled"}, False];
    SetPropertyValue[{"openFileDialog", "fileSelectionMode"}, 
      PropertyValue[{"openFileDialog", "Files_Only"}]];
    ];
  returnValue = InvokeMethod[{"openFileDialog", "showOpenDialog"}, WidgetReference["wizardFrame"]];
  If[returnValue === PropertyValue[{"openFileDialog", "Approve_Option"}], 
    SetPropertyValue[{"filenameTextField", "text"}, 
      PropertyValue[{PropertyValue[{"openFileDialog", "selectedFile"}], "path"}] ]
    ];
  ];
  
ValidateFileSourcePage[] := Module[{txt, isValid = False},
  txt = PropertyValue[{"filenameTextField", "text"}];
  If[ TrueQ[txt =!= ""] && (FileType[txt] === File),
    isValid = True;
    $wizardSource = {File, txt}, 
    $wizardSource = Null;
    ];
  SetPropertyValue[{"fileSourcePage", "allowNext"}, isValid];
  ];
   
ValidateTextSourcePage[] := Module[{txt, isValid = False},
  txt = PropertyValue[{"textSourceArea", "text"}];
  If[ TrueQ[txt =!= ""],
    isValid = True;
    $wizardSource = {Text, txt},
    $wizardSource = Null;
    ];
  SetPropertyValue[{"textSourcePage", "allowNext"}, isValid];
  ];
  

UpdatePreviewTable[] := Module[{result, rowSeps = {}, colSeps = {},r,c,ignore},
  rowSeps = Flatten[{"\r", "\n"}];
  colSeps = Flatten[{
    If[ TrueQ[ PropertyValue[{"tabCheckBox", "selected"}]], "\t", {}],
    If[ TrueQ[ PropertyValue[{"commaCheckBox", "selected"}]], ",", {}],
    If[ TrueQ[ PropertyValue[{"spaceCheckBox", "selected"}]], " ", {}],
    If[ TrueQ[ PropertyValue[{"otherCheckBox", "selected"}]] &&
        StringLength[PropertyValue[{"otherTextField", "text"}]] > 0, 
       PropertyValue[{"otherTextField", "text"}], {}]
    }];
  If[ $wizardSource === Null, 
    SetPropertyValue[{"tablePreviewModel", "columnCount"}, 0];
    SetPropertyValue[{"tablePreviewModel", "items"}, Null];
    Return[];
    ];
  ignore = TrueQ[ PropertyValue[{"emptylineCheckBox", "selected"}]];
  If[ MatchQ[$wizardSource, {File, _}],
     result = Import[ Last[$wizardSource], "Table",
     				"LineSeparators" -> rowSeps, 
     				"FieldSeparators" -> colSeps,
     				IgnoreEmptyLines -> ignore];
     ];
  If[ MatchQ[$wizardSource, {Text, _}],
     result = ImportString[ Last[$wizardSource], "Table",
       				"LineSeparators" -> rowSeps, 
       				"FieldSeparators" -> colSeps,
       				IgnoreEmptyLines -> ignore];
     ];
  r = Length[ result];
  c = Max[ Map[ Length, result]];
  result = PadRight[ result,{r,c},""];
  If[ MatrixQ[result],
    SetPropertyValue[{"tablePreviewModel", "columnCount"}, Length[First[result]]];
    SetPropertyValue[{"tablePreviewModel", "items"}, result],
    SetPropertyValue[{"tablePreviewModel", "columnCount"}, 0];
    SetPropertyValue[{"tablePreviewModel", "items"}, Null];
    ];
  ];
