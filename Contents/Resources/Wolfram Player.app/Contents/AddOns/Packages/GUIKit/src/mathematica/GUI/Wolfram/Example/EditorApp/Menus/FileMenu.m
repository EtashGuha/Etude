Widget["Menu", {
  "text" -> "File",
  
  Script[
    uniqueFileIndex = 0;
    
		BindEvent[{"contentTargetDocument", "document"},
      Script[ 
        InvokeMethod[{"contentTarget", "putClientProperty"}, "isDirty", "True"]; 
        fileName = InvokeMethod[{"contentTarget", "getClientProperty"}, "fileName"];
        SetPropertyValue[{"frame", "title"}, "Text Editor - " <> fileName <> " *"];
        ]
      ];
              
    CreateNewFile[] := Module[{fileName},
			fileName = "Untitled-" <> ToString[++uniqueFileIndex];
			SetPropertyValue[{"contentTarget", "text"}, ""];
			InvokeMethod[{"contentTarget", "putClientProperty"}, "isDirty", "False"];
			InvokeMethod[{"contentTarget", "putClientProperty"}, "fileName", fileName];
			SetPropertyValue[{"frame", "title"}, "Text Editor - " <> fileName];
			InvokeMethod[{"contentTarget", "requestFocus"}];
			];
       
    SaveFile[] := Module[{file, path, returnValue},
			Widget["FileDialog", Name -> "saveFileDialog"];
		  returnValue = InvokeMethod[{"saveFileDialog", "showSaveDialog"}, WidgetReference["frame"]];
		  If[ returnValue == PropertyValue[{"saveFileDialog", "APPROVE_OPTION"}],
		     file = PropertyValue[{"saveFileDialog", "selectedFile"}];
		     path = PropertyValue[{file, "path"}];
		     SetPropertyValue[{"frame", "title"}, "Text Editor - " <> path];
				 InvokeMethod[{"contentTarget", "putClientProperty"}, "fileName", path];
		     InvokeMethod[{"contentTarget", "putClientProperty"}, "isDirty", "False"];
		     InvokeMethod[{"class:com.wolfram.guikit.util.FileUtils", "writeTextFile"}, 
		        PropertyValue[{"contentTarget", "text"}],
		        file ];
		     InvokeMethod[{"contentTarget", "requestFocus"}];
         ];
      ];
      
    OpenFile[] := Module[{file, returnValue, path},
				Widget["FileDialog", Name -> "openFileDialog"];
				returnValue = InvokeMethod[{"openFileDialog", "showOpenDialog"}, WidgetReference["frame"]];
				If[ returnValue == PropertyValue[{"openFileDialog", "APPROVE_OPTION"}],
					file = PropertyValue[{"openFileDialog", "selectedFile"}];
					SetPropertyValue[{"contentTarget", "text"},
						InvokeMethod[{"class:com.wolfram.guikit.util.FileUtils", "readTextFile"}, file] ];
					path = PropertyValue[{file, "path"}];
					SetPropertyValue[{"frame", "title"}, "Text Editor - " <> path];
					InvokeMethod[{"contentTarget", "putClientProperty"}, "fileName", path];
					InvokeMethod[{"contentTarget", "putClientProperty"}, "isDirty", "False"];
					InvokeMethod[{"contentTarget", "requestFocus"}];
					];
      	];
      
    CheckSaveCurrentFile[] := Module[{result = True},
       If[ StringMatchQ[ InvokeMethod[{"contentTarget", "getClientProperty"}, "isDirty"], "True"], 
         processCurrent = Abort;
				 InvokeMethod[{"currentFileDialog", "show"}];
				 If[ processCurrent === True, SaveFile[]];      
				 If[ processCurrent === Abort, result = False];
         ];
       result
       ];
       
    CreateNewFile[];
    ],
    
  Widget["Dialog", {
		"title" -> "Text Editor",
		"modal" -> True,
		"resizable" -> False,
	  Widget["Panel", {
	   Widget["Label", {"text"->"Save modified current document?"}, Name -> "currentFileLabel"],
	   WidgetSpace[10],
	   {WidgetFill[], 
	    Widget["Button",{"text"->" Yes  ",
	      BindEvent["action", 
	        Script[
	        processCurrent = True;
	        InvokeMethod[{"currentFileDialog", "dispose"}];
	        ]]}],
	    WidgetSpace[5],
	    Widget["Button",{"text"->"  No  ",
	      BindEvent["action", 
	        Script[
	        processCurrent = False;
	        InvokeMethod[{"currentFileDialog", "dispose"}];
	        ]]}], 
	    WidgetSpace[5],
	    Widget["Button",{"text"->"Cancel",
	      BindEvent["action", 
	        Script[
	        processCurrent = Abort;
	        InvokeMethod[{"currentFileDialog", "dispose"}];
	        ]]}],
	    WidgetFill[]}
	    },
	    WidgetLayout->{"Border"->{5,5,5,5}}],
	 
		InvokeMethod["center"]
		}, Name -> "currentFileDialog",
		InitialArguments -> {WidgetReference["frame"]}],
    
  Widget["MenuItem", {
    "text" -> "New",
    "accelerator" -> Widget["MenuShortcut", InitialArguments -> {"N"}],
    BindEvent["action", 
      Script[
        If[ CheckSaveCurrentFile[],
          CreateNewFile[];
          ];
        ]
      ]
    }], 

  Widget["MenuItem", {
    "text" -> "Open",
    "accelerator" -> Widget["MenuShortcut", InitialArguments -> {"O"}],
    BindEvent["action", 
      Script[
        If[ CheckSaveCurrentFile[],
          OpenFile[];
          ];
        ]
      ]
    }], 

  Widget["MenuItem", {
    "text" -> "Close",
    BindEvent["action", 
      Script[
        If[ CheckSaveCurrentFile[],
          CreateNewFile[];
          ];
        ]
      ]
    }], 
    
  Widget["MenuSeparator"], 

  Widget["MenuItem", {
      "text" -> "Save",
      "accelerator" -> Widget["MenuShortcut", InitialArguments -> {"S"}],
      BindEvent["action", 
        Script[
          SaveFile[];
          ]
        ]
      }],

  Widget["MenuSeparator"], 

  Widget["MenuItem", {
    "text" -> "Exit",
    BindEvent["action", 
      Script[
        If[ CheckSaveCurrentFile[],
          CloseGUIObject[];
          ];
        ]
      ]}]
  }]

