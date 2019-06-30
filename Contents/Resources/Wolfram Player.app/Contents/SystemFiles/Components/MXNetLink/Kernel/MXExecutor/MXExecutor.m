Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["MXExecutor"]

SetUsage @ "
MXExecutor[id$] represents an MXExecutor managed by MXNet."

(******************************************************************************)

PackageExport["MXExecutorData"]

SetUsage @ "
MXExecutorData[$$] is a wrapper around an MXExecutor that also stores the \
arrays associated with the executor."
