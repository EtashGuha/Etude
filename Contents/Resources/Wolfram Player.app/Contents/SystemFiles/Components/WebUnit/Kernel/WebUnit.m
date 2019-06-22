(* ::Package:: *)

BeginPackage["WebUnit`", {"PacletManager`"}]


System`StartWebSession;
System`WebExecute;
System`WebSessionObject;
System`$CurrentWebSession;
System`WebSessions;
System`WebImage;
System`WebWindowObject;
System`WebElementObject;


WebUnit`$Link;
Unprotect[{ StartWebSession, WebExecute, WebSessionObject, WebSessions, WebImage, WebWindowObject, WebElementObject }];

Get[ FileNameJoin[{DirectoryName[$InputFileName], "Usage.m"}] ];
Get[ FileNameJoin[{DirectoryName[$InputFileName], "errorMessages.m"}] ];



Begin["`Private`"]
(* Implementation of the package *)
coreFiles = {"Utilities.m","WebDriverAPI.m"};

Get@FileNameJoin[{DirectoryName[$InputFileName], #}]&/@coreFiles;

Module[{functionFiles},

  functionFiles = FileNames@FileNameJoin[{DirectoryName[$InputFileName], "wd*.m"}];
  Get[#]&/@functionFiles;
];

Protect[{ StartWebSession, WebExecute, WebSessionObject, WebSessions, WebImage, WebWindowObject, WebElementObject }];
End[];
EndPackage[];

