(* ::Package:: *)

Begin["System`ConvertersDump`"]


ImportExport`RegisterExport[
  "GIF",
  System`Convert`CommonGraphicsDump`ExportElementsToRasterFormat["GIF", ##]&,
  "Sources" -> {"Convert`CommonGraphics`"},
  "DefaultElement" -> Automatic,
  "Options" -> SortBy[{"AnimationDuration", "AnimationRepetitions", Background, "Comments", "ControlAppearance", "DisplayDurations", Dithering, "FrameRate", ImageSize, "Interlaced", "PreserveManipulateInitialization", "TransparentColor"}, ToString],
  "BinaryFormat" -> True
]


End[]

