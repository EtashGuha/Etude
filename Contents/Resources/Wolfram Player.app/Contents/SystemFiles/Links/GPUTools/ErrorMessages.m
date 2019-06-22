
(* ::Section:: *)
(* Hyperlink in Messages *)

GPUTools`Message`MakeDocumentationLink[text_String, hyperlink_String] :=
	"\!\(\*\nButtonBox[\"" <> text <> "\",ButtonStyle->\"Link\",ButtonData:>\"paclet:" <> hyperlink <> "\"]\)"

GPUTools`Message`GPUSystemRequirements["CUDALink" | "CUDA"] =
	GPUTools`Message`MakeDocumentationLink["CUDALink System Requirements", "CUDALink/tutorial/Setup"]
GPUTools`Message`GPUSystemRequirements["OpenCLLink" | "OpenCL"] =
	GPUTools`Message`MakeDocumentationLink["OpenCLLink System Requirements", "OpenCLLink/tutorial/Setup"]

GPUTools`Message`ReferToSystemRequirements[api_] := GPUTools`Message`ReferToSystemRequirements[api] =
	"Refer to " <> GPUTools`Message`GPUSystemRequirements[api] <> " for system requirements."

(* ::Section:: *)
(* Failure Reasons *)

GPUTools`Message`nodev[api_String] := "No " <> api <> " devices detected on system. " <> GPUTools`Message`ReferToSystemRequirements[api]
GPUTools`Message`nolib[api_String] := "The " <> api <> " library `1` was not found in path. Installation might be corrupt."
GPUTools`Message`init[api_String] := api <> " failed to initialize. " <> GPUTools`Message`ReferToSystemRequirements[api]
GPUTools`Message`invsys[api_String] := api <> " is not supported on `1`. " <> GPUTools`Message`ReferToSystemRequirements[api]
GPUTools`Message`legcy[api_String] := api <> " is no longer supported on `1`. " <> GPUTools`Message`ReferToSystemRequirements[api]
GPUTools`Message`invdevnm[api_String] := api <> " is not supported on device `1`. " <> GPUTools`Message`ReferToSystemRequirements[api]
GPUTools`Message`invdrv[api_String] := api <> " was not able to find a valid " <> api <> " driver. " <> GPUTools`Message`ReferToSystemRequirements[api]
GPUTools`Message`invdrvp[api_String] := api <> " was not able to find a valid " <> api <> " driver in `1`. " <> GPUTools`Message`ReferToSystemRequirements[api]
GPUTools`Message`invdrivver[api_String] :=  api <> " was unable to determine the " <> api <> " driver version installed on the system. " <> GPUTools`Message`ReferToSystemRequirements[api]
GPUTools`Message`invdrivverv[api_String] := api <> " is not supported using " <> api <> " driver version `1` installed on the system. " <> GPUTools`Message`ReferToSystemRequirements[api]
GPUTools`Message`invdrivverd[api_String] := api <> " was unable to determine the " <> api <> " driver version installed on the system in `1`. " <> GPUTools`Message`ReferToSystemRequirements[api]
GPUTools`Message`invclib[api_String] := api <> " was unable to find the " <> api <> " libraries. " <> GPUTools`Message`ReferToSystemRequirements[api]
GPUTools`Message`invclibp[api_String] := api <> " was unable to find the " <> api <> " libraries in `1`. " <> GPUTools`Message`ReferToSystemRequirements[api]
GPUTools`Message`invcudaver[api_String] := api <> " was unable to determine the " <> api <> " version installed on the system. " <> GPUTools`Message`ReferToSystemRequirements[api]
GPUTools`Message`invcudaverv[api_String] := api <> " is not supported using " <> api <> " version `1` installed on the system. " <> GPUTools`Message`ReferToSystemRequirements[api]
GPUTools`Message`invcudaverd[api_String] := api <> " was unable to determine the " <> api <> " version installed on the system in `1`. " <> GPUTools`Message`ReferToSystemRequirements[api]
GPUTools`Message`invtkver[api_String] := api <> " was unable to determine the toolkit version installed on the system. " <> GPUTools`Message`ReferToSystemRequirements[api]
GPUTools`Message`invtkverv[api_String] := "Toolkit version `1` is not supported by " <> api <> ". " <> GPUTools`Message`ReferToSystemRequirements[api]
GPUTools`Message`syslibfld[api_String] := api <> " failed to load system libraries. " <> GPUTools`Message`ReferToSystemRequirements[api]
GPUTools`Message`libldfld[api_String] := api <> " failed to load libraries. " <> GPUTools`Message`ReferToSystemRequirements[api]
GPUTools`Message`gpures[api_String] := "One or more " <> api <> " resource failed to load. " <> GPUTools`Message`ReferToSystemRequirements[api]
