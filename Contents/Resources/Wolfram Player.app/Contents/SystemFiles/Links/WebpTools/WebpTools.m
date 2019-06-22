BeginPackage["WebpTools`"]
Begin["`Private`"]

InitWebpTools[] := 
  Module[{dir, dlls, WebpToolsdll, DLLTable}, 
   DLLTable = {"MacOSX-x86-64" -> \
{FileNameJoin[{$InstallationDirectory, "SystemFiles", "Links", 
         "WebpTools", "LibraryResources", 
         "MacOSX-x86-64"}], {"libwebp.dylib", "libwebpdecoder.dylib", "libwebpdemux.dylib", "libwebpmux.dylib"}}, 
     "Linux" -> {FileNameJoin[{$InstallationDirectory, "SystemFiles", 
         "Links", "WebpTools", "LibraryResources", 
         "Linux"}], {"libwebp.so", "libwebpdecoder.so", "libwebpdemux.so", "libwebpmux.so"}}, 
     "Linux-x86-64" -> {FileNameJoin[{$InstallationDirectory, 
         "SystemFiles", "Links", "WebpTools", "LibraryResources", 
         "Linux-x86-64"}], {"libwebp.so", "libwebpdecoder.so", "libwebpdemux.so", "libwebpmux.so"}}, 
     "Linux-ARM" -> {FileNameJoin[{$InstallationDirectory, "SystemFiles", 
         "Links", "WebpTools", "LibraryResources", 
         "Linux-ARM"}], {"libwebp.so", "libwebpdecoder.so", "libwebpdemux.so", "libwebpmux.so"}}, 
     "Windows" -> {FileNameJoin[{$InstallationDirectory, 
         "SystemFiles", "Links", "WebpTools", "LibraryResources", 
         "Windows"}], {"msvcr110.dll", "libwebp.dll", "libwebpdecoder.dll", "libwebpdemux.dll", "libwebpmux.dll"}}, 
     "Windows-x86-64" -> {FileNameJoin[{$InstallationDirectory, 
         "SystemFiles", "Links", "WebpTools", "LibraryResources", 
         "Windows-x86-64"}], {"msvcr110.dll", "libwebp.dll", "libwebpdecoder.dll", "libwebpdemux.dll", "libwebpmux.dll"}}};
   dlls = Cases[DLLTable, ($SystemID -> l_) :> l];
   If[dlls === {}, Message[WebpTools::sys, "Incompatible SystemID"];
    $InitWebpTools = $Failed;
    Return[$Failed]];
   {dir, dlls} = First@dlls;
   Needs["CCompilerDriver`"];
   LibraryLoad[FileNameJoin[{dir, #}]] & /@ dlls;
   WebpToolsdll = FileNameJoin[{dir, "WebpTools"}];
   $ReadImageWebpFun = LibraryFunctionLoad[WebpToolsdll, "ReadImageWebp", {"UTF8String"}, {"Image"}];
   $ReadICCWebpFun = LibraryFunctionLoad[WebpToolsdll, "ReadICCWebp", {"UTF8String"}, {"RawArray"}];
   $WriteImageWebpFun = LibraryFunctionLoad[WebpToolsdll, "WriteImageWebp", {{"UTF8String"}, {"Image"}, _Integer}, "UTF8String"];
   $ReadMetadataWebpFun = LibraryFunctionLoad[WebpToolsdll, "ReadMetadataWebp", {{"UTF8String"}, {"UTF8String"}}, "UTF8String"];
   $ReadImageSizeWebpFun = LibraryFunctionLoad[WebpToolsdll, "ReadImageSizeWebp", {{"UTF8String"}, {"UTF8String"}}, _Integer];
   $InitWebpTools = If[$ReadImageWebpFun === $Failed || $WriteImageWebpFun === $Failed || $ReadICCWebpFun === $Failed || $ReadMetadataWebpFun === $Failed || $ReadImageSizeWebpFun === $Failed, $Failed, True];
   $InitWebpTools];

End[]
EndPackage[]
