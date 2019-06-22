Widget["Frame", {
  "title"->"TerraService", 

  {{Widget["ScrollPane", {
    "viewportView" -> 
      Widget["Panel", {
        "preferredSize"->Widget["Dimension", {"width" -> 150, "height" -> 600}],
        Widget["Panel", {
          {
            Widget["TextField", {
              BindEvent["action", Script[update["SEARCH"]], InvokeThread->"New"]
            }, Name->"placeField"],
            Widget["Button", {
              "icon"-> Widget["Icon", {"path" -> "TerraService/icons/Play16.gif"}],
              "preferredSize"->Widget["Dimension", {"width" -> 17, "height" -> 17}],
              BindEvent["action", Script[update["SEARCH"]], InvokeThread->"New"]
            }]
          }
        }, WidgetLayout->{"Border"->"Search"}], 
        Widget["Panel", {
          Widget["ComboBox", {
            "items"->{"Aerial Photo", "Topo Map"}, 
            BindEvent["action", Script[update["VIEW"]], InvokeThread->"New"]
          }, Name->"themeChoice"]
        }, WidgetLayout->{"Border"->"View", "Stretching"->{True,False}}],
        Widget["Panel", {
          WidgetGroup[{
            WidgetFill[], 
            Widget["Button", {
              "icon" -> Widget["Icon", {"path" -> "TerraService/icons/NorthWest24.gif"}],
              "preferredSize"->Widget["Dimension", {"width" -> 26, "height" -> 26}], 
              "toolTipText" -> "Move north west.",
              BindEvent["action", Script[moveNorthWest[]], InvokeThread->"New"]}], 
            Widget["Button", {
              "icon" -> Widget["Icon", {"path" -> "TerraService/icons/North24.gif"}],
              "preferredSize"->Widget["Dimension", {"width" -> 26, "height" -> 26}], 
              "toolTipText" -> "Move north.",
              BindEvent["action", Script[moveNorth[]], InvokeThread->"New"]}], 
            Widget["Button", {
              "icon" -> Widget["Icon", {"path" -> "TerraService/icons/NorthEast24.gif"}],
              "preferredSize"->Widget["Dimension", {"width" -> 26, "height" -> 26}], 
              "toolTipText" -> "Move north east.",
              BindEvent["action", Script[moveNorthEast[]], InvokeThread->"New"]}],
            WidgetFill[]
          },WidgetLayout->{"Spacing"->0}],
          WidgetGroup[{
            WidgetFill[], 
            Widget["Button", {
              "icon" -> Widget["Icon", {"path" -> "TerraService/icons/West24.gif"}],
              "preferredSize"->Widget["Dimension", {"width" -> 26, "height" -> 26}], 
              "toolTipText" -> "Move west.",
              BindEvent["action", Script[moveWest[]], InvokeThread->"New"]}], 
            Widget["Button", {
              "preferredSize"->Widget["Dimension", {"width" -> 26, "height" -> 26}], 
              "icon" -> Widget["Icon", {"path" -> "TerraService/icons/Empty24.gif"}]
            }], 
            Widget["Button", {
              "icon" -> Widget["Icon", {"path" -> "TerraService/icons/East24.gif"}],
              "preferredSize"->Widget["Dimension", {"width" -> 26, "height" -> 26}], 
              "toolTipText" -> "Move east.",
              BindEvent["action", Script[moveEast[]], InvokeThread->"New"]}],
            WidgetFill[]
          },WidgetLayout->{"Spacing"->0}],
          WidgetGroup[{
            WidgetFill[], 
            Widget["Button", {
              "icon" -> Widget["Icon", {"path" -> "TerraService/icons/SouthWest24.gif"}],
              "preferredSize"->Widget["Dimension", {"width" -> 26, "height" -> 26}], 
              "toolTipText" -> "Move south west.",
              BindEvent["action", Script[moveSouthWest[]], InvokeThread->"New"]}], 
            Widget["Button", {
              "icon" -> Widget["Icon", {"path" -> "TerraService/icons/South24.gif"}],
              "preferredSize"->Widget["Dimension", {"width" -> 26, "height" -> 26}], 
              "toolTipText" -> "Move south.",
              BindEvent["action", Script[moveSouth[]], InvokeThread->"New"]}], 
            Widget["Button", {
              "icon" -> Widget["Icon", {"path" -> "TerraService/icons/SouthEast24.gif"}],
              "preferredSize"->Widget["Dimension", {"width" -> 26, "height" -> 26}], 
              "toolTipText" -> "Move south east.",
              BindEvent["action", Script[moveSouthEast[]], InvokeThread->"New"]}],
            WidgetFill[]
          },WidgetLayout->{"Spacing"->0}]
        },WidgetLayout->{"Border"->"Navigate"}],
        Widget["Panel", {
          Widget["Slider", {
            "minimum"->1, 
            "maximum"->7,
            "orientation"->PropertyValue["VERTICAL"],
            "paintLabels"->True, 
            "snapToTicks"->True,
            Widget["class:java.util.Hashtable", Name->"nameTable"],
            Script[
              InvokeMethod[{"nameTable", "put"}, MakeJavaObject[7], Widget["Label", {"text"->"64 meter"}]];
              InvokeMethod[{"nameTable", "put"}, MakeJavaObject[6], Widget["Label", {"text"->"32 meter"}]];
              InvokeMethod[{"nameTable", "put"}, MakeJavaObject[5], Widget["Label", {"text"->"16 meter"}]];
              InvokeMethod[{"nameTable", "put"}, MakeJavaObject[4], Widget["Label", {"text"->"8 meter"}]];
              InvokeMethod[{"nameTable", "put"}, MakeJavaObject[3], Widget["Label", {"text"->"4 meter"}]];
              InvokeMethod[{"nameTable", "put"}, MakeJavaObject[2], Widget["Label", {"text"->"2 meter"}]];
              InvokeMethod[{"nameTable", "put"}, MakeJavaObject[1], Widget["Label", {"text"->"1 meter"}]];
              SetPropertyValue[{"scaleChoice", "labelTable"}, WidgetReference["nameTable"]];
            ],
            BindEvent["mouseReleased", Script[update["SCALE"]], InvokeThread->"New"]
          }, Name->"scaleChoice"]
        }, WidgetLayout->{"Border"->"Zoom", "Stretching"->{True,False}}], 
        Widget["Panel", {
          Widget["Panel", {
            Widget["TextField", Name->"latField"]
          }, WidgetLayout->{"Border"->"Latitude"}], 
          Widget["Panel",{
            Widget["TextField", Name->"lonField"]
          }, WidgetLayout->{"Border"->"Longitude"}], 
          Widget["Button",{
            "text"->"GO",
            BindEvent["action", Script[update["LATLON"]], InvokeThread->"New"]}]
        }, WidgetLayout->{"Border"->""}], 
        WidgetFill[]     
      }]
  }]},

  {Widget["ScrollPane", {
    "viewportView" -> 
      Widget["Panel", {
        {
          Widget["Label", {"text"->""}, Name->"location", WidgetLayout->{"Border"->"Location", "Stretching"->{True,False}}],
          Widget["Label", {"text"->""}, Name->"capture", WidgetLayout->{"Border"->"Date Captured", "Stretching"->{True,False}}]
        }, 
        Widget["Table", {
          "columnEditable"->{False},
          "rowMargin"->0,
          PropertyValue["selectionModel", Name -> "imageTableSelectionModel"],
          BindEvent[{"imageTableSelectionModel", "valueChanged"}, 
            Script[
              Block[{row = PropertyValue[{"imageTable", "selectedRow"}], 
                     col = PropertyValue[{"imageTable", "selectedColumn"}]}, 
                Which[
                  row == 0 && col == 0, moveNorthWest[], 
                  row == 0 && col == 1, moveNorth[], 
                  row == 0 && col == 2, moveNorthEast[], 
                  row == 1 && col == 0, moveWest[], 
                  row == 1 && col == 2, moveEast[], 
                  row == 2 && col == 0, moveSouthWest[], 
                  row == 2 && col == 1, moveSouth[], 
                  row == 2 && col == 2, moveSouthEast[]
                ]
              ];
              InvokeMethod[{"imageTable", "changeSelection"}, 1, 1, False, False];
            ]
          ],
          PropertyValue["columnModel", Name->"columnModel"],
          SetPropertyValue[{"columnModel", "columnMargin"}, 0],
          "preferredSize"->Widget["Dimension", {"width" -> 600, "height" -> 600}]
        }, Name->"imageTable", WidgetLayout->{"Stretching"->{False, False}}],
        "preferredSize"->Widget["Dimension", {"width" -> 600, "height" -> 650}]
      }]
  }]}}, 
  Widget["Panel", {
    WidgetGroup[{
      Widget["Label", Name->"status"]
    }, WidgetLayout->{"Border"->3}]
  }],

  Script[
    Needs["WebServices`"];
  ], 
  Script[
    updateStatus[msg_String] :=
      Module[{},
        SetPropertyValue[{"status", "text"}, msg];
        ObjectReference["status"]@repaint[200];
      ];
  ], 
  Script[
    installed = InstallService["http://terraservice.net/TerraService.asmx?WSDL"];
    If[installed === $Failed, updateStatus["Failed to install service."]];
  ], 
  
  Script[

    getImage[id_List] := getImage[id] = 
      Block[{native = InvokeServiceOperation::native, 
             rspns = InvokeServiceOperation::rspns, 
             result, 
             icon},
        Off[InvokeServiceOperation::native];
        Off[InvokeServiceOperation::rspns];
        result = GetTile[TileId@@id, Timeout->10000];
        If[result === $Failed, updateStatus["Problem creating image."]; error = True; result = Null];
        If[Head[native] =!= $Off, On[InvokeServiceOperation::native]];
      	If[Head[rspns] =!= $Off, On[InvokeServiceOperation::rspns]];
      	icon = Widget["Icon", InitialArguments->{}];
      	If[result =!= Null, 
      	  SetPropertyValue[{icon, "data"}, FromCharacterCode[First[result]]];
      	];
      	KeepJavaObject[icon];
      	icon
      ];      
    
    getMetaFromLonLat[lat_, lon_, theme_String, scale_String] := 
      getMetaFromLonLat[lat, lon, theme, scale] =
        Block[{native = InvokeServiceOperation::native, 
               rspns = InvokeServiceOperation::rspns, 
               result},
          Off[InvokeServiceOperation::native];
          Off[InvokeServiceOperation::rspns];
          result =      
            GetTileMetaFromLonLatPt[LonLatPt["Lon" -> longitude, "Lat" -> latitude], theme, scale, Timeout->10000];
          If[Head[native] =!= $Off, On[InvokeServiceOperation::native]];
    	    If[Head[rspns] =!= $Off, On[InvokeServiceOperation::rspns]];
      	  result
        ];

    getPlace[plc_String] := getPlace[plc] = 
      Block[{native = InvokeServiceOperation::native, 
             rspns = InvokeServiceOperation::rspns, 
             result},
        Off[InvokeServiceOperation::native];
        Off[InvokeServiceOperation::rspns];
        result = GetPlaceList[plc, 1, True, Timeout->10000];
        If[Head[native] =!= $Off, On[InvokeServiceOperation::native]];
      	If[Head[rspns] =!= $Off, On[InvokeServiceOperation::rspns]];
      	result
      ];

    lonLatToPlace[lon_, lat_] := 
      lonLatToPlace[lon, lat] = 
        Block[{native = InvokeServiceOperation::native, 
               rspns = InvokeServiceOperation::rspns, 
               result},
          Off[InvokeServiceOperation::native];
          Off[InvokeServiceOperation::rspns];
          result =      
            ConvertLonLatPtToNearestPlace[LonLatPt["Lon" -> lon, "Lat" -> lat], Timeout->10000];
          If[Head[native] =!= $Off, On[InvokeServiceOperation::native]];
      	  If[Head[rspns] =!= $Off, On[InvokeServiceOperation::rspns]];
    	    result
        ];

    getMetaFromTileId[t_String, scl_String, scn_, x1_, y1_] := 
      getMetaFromTileId[t, scl, scn, x1, y1]  = 
      
        Block[{native = InvokeServiceOperation::native, 
               rspns = InvokeServiceOperation::rspns, 
               result},
          Off[InvokeServiceOperation::native];
          Off[InvokeServiceOperation::rspns];
          result =      
            GetTileMetaFromTileId[TileId["Theme" -> t, "Scale" -> scl, "Scene" -> scn, "X" -> x1, "Y" -> y1], Timeout->10000];
          If[Head[native] =!= $Off, On[InvokeServiceOperation::native]];
    	    If[Head[rspns] =!= $Off, On[InvokeServiceOperation::rspns]];
      	  result
        ];

    updateMeta[] :=
      Module[{meta, capture},
        meta = getMetaFromLonLat[latitude, longitude, theme, scale];
        If[meta === $Failed, 
          updateStatus["Failed to retrieve meta data for this location."];
          Return[$Failed]
        ];
        capture = meta["Capture"];
        capture = capture /. XMLSchema`SchemaDateTime[ a__] :> DateString[{a}];
        SetPropertyValue[{"capture", "text"}, capture];
        x = meta["Id"]["X"];
        y = meta["Id"]["Y"];
        scene = meta["Id"]["Scene"];
      ];

    setPlace[p_String] :=
      Module[{result, latLon, plc},
        result = getPlace[p];
        If[result =!= $Failed && result =!= Null && result =!= {},
          latLon = First[result]["Center"];
          latitude = latLon["Lat"];
          longitude = latLon["Lon"];
          plc = First[result]["Place"];
          location = plc["City"] <> ", " <> plc["State"] <> ", " <> plc["Country"],
          If[result === {}, 
            updateStatus["Place not found: " <> p],
            updateStatus["Failed to retrieve data for " <> p];
          ];
          Return[$Failed];
        ];
        location
      ];

    setLatLon[lat_, lon_] := 
      Module[{result},
        latitude = lat;
        longitude = lon;
        result = lonLatToPlace[longitude, latitude];
        If[result === $Failed || result === Null || result === {}, 
          updateStatus["Failed to retrieve place data for (" <> ToString[longitude] <> ", " <> ToString[latitude] <> ")"];
          Return[$Failed],
          location = result;
        ];
      ];

    setScale[s_Integer] := (
      scale = scales[[s]];
    );

    setTheme[t_String] := (
      theme = t;
    ); 

    updateXY[] :=
      Module[{meta},
        meta = getMetaFromTileId[theme, scale, scene, x, y];
        If[meta === $Failed, updateStatus["Failed to retrieve meta data for this location."];Return[]];
        SetPropertyValue[{"capture", "text"}, meta["Capture"]];        
        latitude = meta["Center"]["Lat"];
        SetPropertyValue[{"latField", "text"}, ToString[latitude]];        
        longitude = meta["Center"]["Lon"];
        SetPropertyValue[{"lonField", "text"}, ToString[longitude]];                
        location = lonLatToPlace[longitude, latitude];
        If[location === $Failed || location === Null || location === {}, 
          updateStatus["Failed to retrieve place data for (" <> longitude <> ", " <> latitude <>")"];
          Return[]
        ];
        SetPropertyValue[{"location", "text"}, location];        
      ];
      
    moveWest[] := (
      x--;
      setImages[];
      updateXY[];
    );

    moveEast[] := (
      x++;
      setImages[];
      updateXY[];
    );

    moveNorth[] := (
      y++;
      setImages[];
      updateXY[];
    );

    moveSouth[] := (
      y--;
      setImages[];
      updateXY[];
    );

    moveNorthWest[] := (
      x--;
      y++;
      setImages[];
      updateXY[];
    );

    moveNorthEast[] := (
      x++;
      y++;
      setImages[];
      updateXY[];
    );

    moveSouthWest[] := (
      x--;
      y--;
      setImages[];
      updateXY[];
    );

    moveSouthEast[] := (
      x++;
      y--;
      setImages[];
      updateXY[];
    );

    scales = {"Scale1m", "Scale2m", "Scale4m", "Scale8m", 
              "Scale16m", "Scale32m", "Scale64m"};
             
    x = 0;
    y = 0;
    scene = 0;
    theme = "Photo";
    scale = "Scale16m";
      
    setPlace["Champaign, IL"];

    setImages[] :=
      Module[{i = 1}, 
       SetPropertyValue[{"imageTable", "items"}, 
         Outer[
           (updateStatus["Downloading image (" <> ToString[i] <> " of 9)"];
            i = i + 1;
            getImage[{"Theme" -> theme, 
                      "Scale" -> scale, 
                       "Scene" -> scene, 
                       "X" -> #2,
                       "Y" -> #1 
                       }]) &, 
           Reverse[Range[y - 1, y + 1]],
           Range[x - 1, x + 1]]];
      ];

    update[function_] := 
      Module[{lat, lon, result, error, place},
        
        If[installed === $Failed, updateStatus["Failed to update.  Service is not installed."];Return[]];
        
        error = False;
        
        (* Update place *)
        Switch[function, 
          "SEARCH", 
          place = PropertyValue[{"placeField", "text"}];
          updateStatus["Searching for " <> place <>"."];
          result = setPlace[PropertyValue[{"placeField", "text"}]];
          If[result === $Failed, resetFields[];Return[]];
          SetPropertyValue[{"location", "text"}, result];
          SetPropertyValue[{"latField", "text"}, ToString[latitude]];
          SetPropertyValue[{"lonField", "text"}, ToString[longitude]],
          "LATLON", 
          lat = ToExpression[PropertyValue[{"latField", "text"}], InputForm, HoldComplete];
          lon = ToExpression[PropertyValue[{"lonField", "text"}], InputForm, HoldComplete];
          If[MatchQ[lat, HoldComplete[_Integer | _Real]] &&
             MatchQ[lon, HoldComplete[_Integer | _Real]],
            result = setLatLon[First[lat], First[lon]];
            If[result === $Failed, 
              Return[],   
              SetPropertyValue[{"location", "text"}, location];         
            ],
            updateStatus["Please input correct coordinates for latitude and longitude."];
            Return[];
          ];
        ];
        (* Update theme *)
        Switch[PropertyValue[{"themeChoice", "selectedItem"}],
          "Aerial Photo", setTheme["Photo"],
          "Topo Map", setTheme["Topo"]];
          
        (* Update scale *)
        If[StringMatchQ[theme, "Topo"] && PropertyValue[{"scaleChoice", "value"}] == 1,
          SetPropertyValue[{"scaleChoice", "value"}, 2];
        ];
        setScale[PropertyValue[{"scaleChoice", "value"}]];
          
        (* Update images *)
        result = updateMeta[];
        If[result === $Failed, Return[]];
        setImages[];
        
        If[!error, updateStatus["Done"]];
      ];

    If[installed === $Failed, updateStatus["Failed to update.  Service is not installed."];Return[]];  
    error = False;
    SetPropertyValue[{"themeChoice", "selectedItem"}, theme];
    SetPropertyValue[{"scaleChoice", "value"}, 5];
    SetPropertyValue[{"location", "text"}, location];
    updateMeta[];
    setImages[];
    SetPropertyValue[{"latField", "text"}, ToString[latitude]];
    SetPropertyValue[{"lonField", "text"}, ToString[longitude]];
    updateStatus["Done"];
    If[!error, updateStatus["Done"]];    
    
  ]
  
}, WidgetLayout->{"Grouping"->Column}]