Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["NDArray"]

SetUsage @ "
NDArray[id$] represents a numeric array managed by MXNet."

(******************************************************************************)

PackageExport["NDArrayQ"]

NDArrayQ[NDArray[_Integer]?System`Private`NoEntryQ] := True
NDArrayQ[_] = False
