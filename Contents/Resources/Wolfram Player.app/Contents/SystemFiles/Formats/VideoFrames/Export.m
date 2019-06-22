(* ::Package:: *)

ImportExport`RegisterExport[
  "VideoFrames",
  System`Convert`VideoFramesDump`ExportVideoFrames,
  "Sources" -> {"Convert`CommonGraphics`", "Convert`VideoFrames`"},
  "FunctionChannels" -> {"FileNames"},
  "OriginalChannel" -> True,
  "ReturnsOutputToUser" -> True,
  "AvailableElements" -> {"Animation", "GraphicsList", "ImageList"}
]
