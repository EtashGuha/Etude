

Begin["DeployedResourceShingle`"]

Begin["`Private`"]

$drsDirectory=DirectoryName[System`Private`$InputFileName];
$webResourcesDirectory=FileNameJoin[{$drsDirectory,"WebResources"}];
$defaultTemplateFile=templateFile["dataresource"];

headerfile[_]:=FileNameJoin[{$webResourcesDirectory,"header.xml"}]

templateFile[rtype_String]:=templatefile[FileNameJoin[{$webResourcesDirectory,ToLowerCase[rtype]<>"shingle.xml"}]]
templatefile[file_]:=file/;FileExistsQ[file]

templateFile[_]:=$defaultTemplateFile
templatefile[_]:=$defaultTemplateFile

getWebResourceInfo[rtype_]:=Association[
		"JSResources"->{
			"https://www.wolframcdn.com/javascript/jquery-2.1.4.min.js",
			"https://www.wolframcloud.com/objects/resourcesystem/webresources/DataRepository/1.1/libs.js", (* include in RSC paclet ? *)
			"https://www.wolframcloud.com/objects/resourcesystem/webresources/DataRepository/1.1/main.js"  (* include in RSC paclet ? *)
			},
		"CSSResources"->{
			"https://fonts.googleapis.com/css?family=Source+Sans+Pro:400,300,300italic,400italic,600,600italic",
			"https://www.wolframcdn.com/css/normalize.css",
			"https://www.wolframcloud.com/objects/resourcesystem/webresources/DataRepository/1.1/main.css"  (* include in RSC paclet ? *)
			},
		"HomepageLogosFolder"->
			"https://www.wolframcloud.com/objects/resourcesystem/webresources/DataRepository/1.1/homepage-logos"  (* include in RSC paclet ? *),
		"FaviconFolder"->
			"https://www.wolframcloud.com/objects/resourcesystem/webresources/DataRepository/1.1/favicon"  (* include in RSC paclet ? *),
		"HeaderInclude"->headerfile[All]
	]


End[]

End[]