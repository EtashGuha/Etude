(* ::Package:: *)

Begin["EmbedCode`Java`Private`"]

$input = $InputFileName;
$root = Nest[ ParentDirectory, DirectoryName[$input], 2 ];
$app = FileNameJoin[ {$root, "Resources", "Android", "WolframCloudApp"} ];

(* Count the number of arguments in the given APIFunction *)
$argc = 0;

(* Line templates to insert code into file templates *)
$inputLine = StringTemplate["baseURL += \"`1`=\" + list.get(`2`) + \"&\";"];
$createEditTextVars = StringTemplate["private EditText field``;"];
$initializeEditTexts = StringTemplate["field`1` = (EditText) findViewById(R.id.field`1`);"];
$getInputsFromFields = StringTemplate["String input`1` = field`1`.getText().toString();"];
$addInputsToList = StringTemplate["list.add(`1`,input`1`);"];
$editTextLayouts = StringTemplate["<EditText
            android:id=\"@+id/field`1`\"
            android:layout_width=\"match_parent\"
            android:layout_height=\"wrap_content\"
            android:layout_margin=\"10dp\"
            android:hint=\"`2`\"
            android:inputType=\"text\" />"];


EmbedCode`Common`iEmbedCode["android", apiFunc_APIFunction, url_, opts:OptionsPattern[]] := Module[ { $tdir, $zip, $zipname },
	(* Display a ProgressIndicator until finished *)
	PrintTemporary[ ProgressIndicator[ Appearance -> "Percolate " ] ];

	(* Get the directory with the template files *)
	$tdir = CreateDirectory[]; 
	DeleteDirectory[$tdir];
	CopyDirectory[ $app, $tdir ];

	(* Determine the number of arguments in apiFunc, used later *)
	$argc = GetArgumentCount[apiFunc];

	(* Insert the new API's URL and other data into the MainActivity.java file *)
	$main = FileNameJoin[ {$tdir, "src", "com", "wolframcloud", "embedcode", "MainActivity.java"} ];
	FileTemplateApply[ FileTemplate[$main], 
	<|
		"url"->url,
		"fields"->CreateEditTextVars[],
		"fieldsInitialization"->InitializeEditTexts[],
		"getInputsFromFields"->GetInputsFromFields[],
		"addInputsToList"->AddInputsToList[]
	|>, $main ];

	(* Insert input variable names into the CloudCompute.java file *)
	$cloud = FileNameJoin[ {$tdir, "src", "com", "wolframcloud", "embedcode", "CloudCompute.java"} ];
	FileTemplateApply[FileTemplate[$cloud],<|"outputWrite"->CreateInputLines[apiFunc]|>,$cloud];

	(* Insert the field layout data into the layout XML file *)
	$layout = FileNameJoin[ {$tdir, "res", "layout", "main.xml"} ];
	FileTemplateApply[FileTemplate[$layout],<|"fields"->CreateTextLayouts[apiFunc]|>,$layout];

	(* Create a zipped archive to be downloaded by user containing all necesarry java files *)
	$zip = CreateArchive[ $tdir ];
	$zipname = Last @ FileNameSplit @ $zip; 
	$zipcloud = CloudObject[ $zipname ];
	CopyFile[ $zip, $zipcloud ];

	(* Pass this Association back to EmbedCode.m to indicate success *)
	<|
	    "EnvironmentName" -> "Android",
	    "FilesSection" -> <| 
	        "FileList" -> { $zipname -> First[$zipcloud] },
	        "Title" -> "Android App",
			"Description" -> "Download and unzip the file below and open with the Android Developer IDE" 
		|>
	|>
]

(* The following functions use the given APIFunction to create Java code and insert it into the above templates *)

CreateInputLines[apiFunc_APIFunction]:=Module[ {$apiInputs, s},
	(* Isolate the list of inputs for the APIFunction *)
	$apiInputs=StringTake[StringSplit[StringTake[ToString[apiFunc, InputForm], {13, -2}], "},"][[1]], {2, -1}];

	(* Isolate the names of the inputs *)
	$apiInputs=Take[StringSplit[$apiInputs, "\""], {1, -1, 4}];
	
	(* Create the java line(s) that write the inputs into the URL *)
	s = "";
	For[i=0,i<$argc,i++,
		s=StringJoin[s, TemplateApply[$inputLine,{$apiInputs[[i+1]],i}] ];
	];
	Return[s];
];

CreateEditTextVars[]:=Module[{s,i},
	s = "";
	For[i=0,i<$argc,i++,
		s=StringJoin[s,TemplateApply[$createEditTextVars,i]];
	];
	Return[s];
];

InitializeEditTexts[]:=Module[{s,i},
	s = "";
	For[i=0,i<$argc,i++,
		s=StringJoin[s,TemplateApply[$initializeEditTexts,i]];
	];
	Return[s];
];

GetInputsFromFields[]:=Module[{s,i},
	s = "";
	For[i=0,i<$argc,i++,
		s=StringJoin[s,TemplateApply[$getInputsFromFields,i]];
	];
	Return[s];
];

AddInputsToList[]:=Module[{s,i},
	s = "";
	For[i=0,i<$argc,i++,
		s=StringJoin[s,TemplateApply[$addInputsToList,i]];
	];
	Return[s];
];

CreateTextLayouts[apiFunc_APIFunction]:=Module[{s,i,apiInputs,apiList},

	(* Isolate the list of input names for the APIFunction *)
	apiInputs=StringTake[StringSplit[StringTake[ToString[apiFunc, InputForm], {13, -2}], "},"][[1]], {2, -1}];
	apiInputs=StringSplit[apiInputs,"\""];
	apiList=Take[apiInputs,{1,-1,4}];

	s = "";
	For[i=0,i<$argc,i++,
		s = StringJoin[s,TemplateApply[$editTextLayouts, { i,apiList[[i+1]] } ]];
	];
	Return[s];
];

GetArgumentCount[apiFunc_APIFunction]:=Module[ {$apiInputs},
	(* Isolate the list of inputs for the APIFunction *)
	$apiInputs=StringTake[StringSplit[StringTake[ToString[apiFunc, InputForm], {13, -2}], "},"][[1]], {2, -1}];

	(* Count the number of arrow operators (and, thus, the number of inputs) *)
	Return[StringCount[$apiInputs,"->"]]
];
										
General::noembedandroid = "EmbedCode for Android only works on objects of type APIFunction. The given object, `1` at `2`, is not of this type."

EmbedCode`Common`iEmbedCode["android ", expr_, uri_, OptionsPattern[]] := (Message[EmbedCode::noembedandroid, expr, uri]; $Failed)

End[]
