Widget["Frame", {
	"title" ->  "Web Service Navigator",
	
	Widget["Panel", {
	 "preferredSize" -> Widget["Dimension", {"width" -> 500, "height" -> 400}],
	 
	{	Widget["TextField", {
			"text"->"http://",
			BindEvent["action", Script[ install[]]]}, Name -> "installField"],
		Widget["Button", {"text" -> "Install",
		  BindEvent["action", Script[ install[]]]
		  }]},
  { Widget["ScrollPane", {
			"viewportView" -> 
			Widget["Tree", {"editable" -> True,
			  "model" -> Widget["TreeModel", {
			     "root" -> Widget["TreeNode", {"userObject" -> "Web Services"}, Name -> "Root"]
			     }, Name -> "treeModel"],
				PropertyValue["selectionModel", Name -> "selectionModel"],
				SetPropertyValue[{"selectionModel", "selectionMode"},  
					PropertyValue[{"selectionModel", "Single_Tree_Selection"}] ]
				}, Name -> "jtree"]
			}],
    {	Widget["Button", {"text"->"Help",
       BindEvent["action", Script[help[]]] }],
			Widget["Button", {"text"->"Uninstall",
       BindEvent["action", Script[uninstall[]]] }],
			Widget["Button", {"text"->"Refresh",
		   BindEvent["action", Script[ refresh[]]] }],               
			WidgetFill[]}
   },
	Widget["Label", {}, Name -> "status"]
	}],

	Script[ Needs["WebServices`"]; ],
  
	Script[
	
    updateStatus[msg_String] := (
			SetPropertyValue[{"status", "text"}, msg];
			InvokeMethod[{"status", "repaint"}, 200]; 
      );
  
    addNodes[nodes_List, parent_] :=
      Module[{node = First[nodes], obj, list = Rest[nodes], childs, child, existingNode = Null, childIndex = 1},         
        childCount = PropertyValue[{parent, "childCount"}];
       	While[existingNode === Null && childIndex <= childCount,
      	  child = PropertyValue[{parent, "childAt", childIndex++}];
      	  If[ PropertyValue[{child, "userObject"}] === node, existingNode = child];
          ];
        If[existingNode === Null,
          obj = Widget["TreeNode"];
          SetPropertyValue[{obj,"userObject"}, node];
          InvokeMethod[{"treeModel", "insertNodeInto"}, obj, parent, PropertyValue[{parent, "childCount"}]],
          obj = existingNode;
          ];
        If[Length[list] > 0, addNodes[list, obj]];
      ];
  
    addService[service_String] :=
      Module[{str = StringToStream[service], nodes, parent},
        nodes = ReadList[str, Word, WordSeparators -> {"`"}];
        addNodes[nodes, WidgetReference["Root"]];
        Close[str];        
        ];
  
    refresh[] := 
      Module[{services},
        updateStatus["Refreshing tree"];
        services = (If[!StringMatchQ[Context[#] <> SymbolName[#], ToString[#]], Context[#] <> ToString[#], ToString[#]] & 
           /@ $InstalledServices);
        addService /@ services;
        InvokeMethod[{"jtree", "updateUI"}];
        updateStatus["Done"];
      ];
      
    install[] := (
        updateStatus["Installing service " <> PropertyValue[{"installField", "text"}]];
        InstallService[PropertyValue[{"installField", "text"}]];
        refresh[];
        updateStatus["Done"];
      );
      
    help[] := 
      Module[{str = "", selectionPath, path, last, leaf = Null, result=Null},
        selectionPath = PropertyValue[{"jtree", "selectionPath"}];
        If[selectionPath === Null, 
          updateStatus["Please select a leaf node to see its help."];
          Return[]
        ];
        path = PropertyValue[{selectionPath, "path"}];
        If[Length[path] > 1, 
          (str = str <> PropertyValue[{#, "userObject"}] <> "`") & /@ Drop[path,1],
          updateStatus["Please select a leaf node to see its help."];
          Return[];
        ];
        last = PropertyValue[{selectionPath, "lastPathComponent"}];
        leaf = PropertyValue[{last, "leaf"}];
        If[leaf === True, 
          str = StringDrop[str, -1];
          result = OperationPalette[ToExpression[str]];
          If[result === Null, 
            updateStatus["Done"],
            updateStatus[result]
          ],
          updateStatus["Please select a leaf node to see its help."];
        ];
        
      ];
      
    uninstallNode[node_] :=
      If[PropertyValue[{node, "leaf"}] === True, InvokeMethod[{"treeModel", "removeNodeFromParent"}, node]];

    uninstall[] := 
      Module[{selectionPath, path, str = "", last, leaf = Null, childs},
        selectionPath = PropertyValue[{"jtree", "selectionPath"}];
        If[selectionPath === Null, 
          updateStatus["Please select a node to uninstall."];
          Return[]
        ];
        path = PropertyValue[{selectionPath, "path"}];
        If[Length[path] > 1, 
          (str = str <> PropertyValue[{#, "userObject"}] <> "`") & /@ Drop[path,1],
          updateStatus["Please select a leaf node to uninstall."];
          Return[];
        ];
        last = PropertyValue[{selectionPath, "lastPathComponent"}];
        leaf = PropertyValue[{last, "leaf"}];
        If[leaf === True, 
          str = StringDrop[str, -1],
          str = str <> "*";
        ];
        UninstallService[str];
        If[leaf === False, 
          childs = InvokeMethod[{last, "children"}];
          For[i=PropertyValue[{last, "childCount"}], i >=1 , i--,
            uninstallNode[PropertyValue[{last, "childAt", i}]];
          ];
        ];
        uninstallNode /@ Reverse[Drop[path, 1]];
        refresh[];
        updateStatus["Done"];
      ];
      
		refresh[];
		]
  
}]