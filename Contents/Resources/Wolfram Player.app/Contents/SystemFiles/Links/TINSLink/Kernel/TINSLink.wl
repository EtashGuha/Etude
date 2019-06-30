(* Wolfram Language Package *)

(* Created by the Wolfram Workbench Aug 4, 2017 *)

BeginPackage["TINSLink`", {"TINSLink`Libraries`", "TINSLink`Utility`"}]
(* Exported symbols added here with SymbolName::usage *) 

StartPacketCapture::usage = "Starts a background worker to capture all packets on an interface.";
StopPacketCapture::usage = "Stops running captures and returns the results.";

System`NetworkPacketTrace::iface = "The \"Interface\" value (`1`) is wrong, it must be a subset of $NetworkInterfaces.";
System`NetworkPacketTrace::pcapfilterstring = "The \"PCAPFilter\" value (`1`) is wrong it must be a valid PCAP filter string."
System`NetworkPacketTrace::noserv = "The service string `1` failed to be interpreted as a \"NetworkService\" Entity."
System`NetworkPacketTrace::noport = "The service `1` doesn't have any any specific ports."
System`NetworkPacketTrace::invalidentity = "The entity `1` isn't a NetworkService entity."

System`NetworkPacketTrace::strinvalid = "The value for the key `1` is invalid, it must be one of `2`."
System`NetworkPacketTrace::portinvalid = "The value for the \"Port\" key is invalid, it must be an Integer greater than zero, list of Integers or a Span[] of increasing integers."
System`NetworkPacketTrace::ipaddrinvalid = "The value for the \"IPAddress\" key is invalid, it must be a valid IPv4 or IPv6 IPAddress."

GetActivePacketCaptures::usage = "Shows all interfaces on which a capture is running";


ImportPacketCapture::usage = "Imports a pcap file and displays the contents.";

CaptureFunction::usage = "Captures all packets sent during execution of a WL function.";

System`NetworkPacketCapture::noserv = "The service string `1` failed to be interpreted as a \"NetworkService\" Entity."
System`NetworkPacketCapture::noport = "The service `1` doesn't have any any specific ports."
System`NetworkPacketCapture::invalidentity = "The entity `1` isn't a NetworkService entity."

System`NetworkPacketCapture::permosx = "Unable to read from network interfaces. Requires read permissions for devices /dev/bpf*."
System`NetworkPacketTrace::permosx = "Unable to read from network interfaces. Requires read permissions for devices /dev/bpf*."
System`NetworkPacketCapture::permlinux = "Unable to read from network interfaces. Requires elevated permissions to capture packets."
System`NetworkPacketTrace::permlinux = "Unable to read from network interfaces. Requires elevated permissions to capture packets."
System`NetworkPacketCapture::permwin = "Unable to read from network interfaces. WinPcap must be installed and running."
System`NetworkPacketTrace::permwin = "Unable to read from network interfaces. WinPcap must be installed and running."

System`NetworkPacketCapture::fereq = "A front end is not available. NetworkPacketCapture requires a front end."

Begin["`Private`"]
(* Implementation of the package *)

(*this is necessary to prevent recursive autoloading of the System` symbols that the paclet manager sets up for us*)
(
  ClearAttributes[#,{Stub,Protected,ReadProtected}];
  Clear[#];
)&/@
  {
    "System`NetworkPacketRecording",
    "System`NetworkPacketTrace",
    "System`NetworkPacketCapture",
    "System`$NetworkInterfaces",
    "System`$DefaultNetworkInterface"
  }


TINSLink`Libraries`initializeLibrariesTINSLink[]

$currentPacketDatasets = {};

nonCrypticToCryptic := Module[{nonCryptics, cryptics},
	nonCryptics = iGetAllInterfaces[];
	cryptics = iGetAllInterfacesCryptic[];
	Association[
		Table[
			Rule[
				nonCryptics[[i]],
				cryptics[[i]]
			],
			{i, 1, Length[nonCryptics]}
		]
	]
];

toCrypticNetworkInterface[interface_List] := Module[{nonCrypToCryp},
		If[!($OperatingSystem === "Windows"),
			Return[interface];
		];
		nonCrypToCryp = nonCrypticToCryptic;
		DeleteMissing[nonCrypToCryp[#1]& /@ interface];
	];

toCrypticNetworkInterface[iface_String] := Module[{},
		If[!($OperatingSystem === "Windows"),
			Return[interface];
		];
		Return[nonCrypticToCryptic[interface] /. Missing[___] -> {}];
	];

(*when we stop a capture we need to append the return value with the current packets so that the gather all function can grab it, so that we can start / stop easily, always adding to the dataset*)
stopCapture[interface_:String|List]:=AppendTo[$currentPacketDatasets,StopPacketCapture[interface]];

(*start capture for now just calls the StartPacketCapture function*)
startCapture[interface_:String|List,filters_?AssociationQ]:= Block[{filterStringSpec=""},
    Which[
      KeyExistsQ[filters,"PCAPFilter"],
      If[StringQ[filters["PCAPFilter"]],
        filterStringSpec = filters["PCAPFilter"]
      ],
      Keys[filters] =!= {},
      filterStringSpec = makeFilterString[KeyDrop[{"PCAPFilter","Interface"}]@filters]/.{"()" -> ""};
    ];
    If[FailureQ[filterStringSpec], filterStringSpec=""];
    StartPacketCapture[interface, filterStringSpec]
]

(*flush capture just deletes all of the previous captures in case it's aborted or something*)
flushCapture[]:= ($currentPacketDatasets = {});

(*get full capture aggregates all the individual captures into a single dataset, then flushes all of the old data for the next time*)
getFullCapture[]:=Block[{res=Dataset[Join@@Normal/@Cases[$currentPacketDatasets,_Dataset]]},flushCapture[];res]

determineInterface[filters_?AssociationQ] := Block[{iface=System`$DefaultNetworkInterface},
	If[KeyExistsQ[filters,"Interface"],
		Which[
			filters["Interface"] === All,
			iface = $NetworkInterfaces,
			MemberQ[$NetworkInterfaces,filters["Interface"]],
			iface = {filters["Interface"]},
			ListQ[filters["Interface"]] && AllTrue[filters["Interface"],StringQ] && SubsetQ[$NetworkInterfaces,filters["Interface"]],
			iface = filters["Interface"]
		]
	];
	iface
]

If[TrueQ[$CloudEvaluation] && !TrueQ[Lookup[CloudSystem`KernelInitialize`$ConfigurationProperties, "AllowNetworkPacketFunctionality"]],
(* Running in cloud environment, define dummy functions that tell the user this functionality is not yet available. *)
	System`NetworkPacketCapture[___] := (Message[General::cloudf, HoldForm@NetworkPacketCapture]; $Failed);
,
(* Else define as usual *)

	(* START: Chart colors *)

	redChart = RGBColor[0.8941, 0.1961, 0.1961];
	blueChart = RGBColor[0.16865, 0.6902, 0.8510];

	(* END: Chart colors *)

	(* START: Icon image assets *)

	stopHover = Graphics[{Thickness[0.058823529411764705`], FaceForm[{RGBColor[0.361, 0.6859999999999999, 1.], Opacity[1.]}], FilledCurve[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}}, {{{15.5, 1.5}, {1.5, 1.5}, {1.5, 15.5}, {15.5, 15.5}}}]}, AspectRatio -> Automatic, ImageSize -> {17., 17.}, PlotRange -> {{0., 17.}, {0., 17.}}];

	stopStatic = Graphics[{Thickness[0.058823529411764705`], FaceForm[{RGBColor[0.537, 0.537, 0.537], Opacity[1.]}], FilledCurve[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}}, {{{15.5, 1.5}, {1.5, 1.5}, {1.5, 15.5}, {15.5, 15.5}}}]}, AspectRatio -> Automatic, ImageSize -> {17., 17.}, PlotRange -> {{0., 17.}, {0., 17.}}];

	pauseHover = Graphics[{Thickness[0.07692307692307693], FaceForm[{RGBColor[0.361, 0.6859999999999999, 1.], Opacity[1.]}], FilledCurve[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}}, {{{1.5, 1.5}, {4.5, 1.5}, {4.5, 17.5}, {1.5, 17.5}}}], FilledCurve[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}}, {{{8.5, 1.5}, {11.5, 1.5}, {11.5, 17.5}, {8.5, 17.5}}}]}, AspectRatio -> Automatic, ImageSize -> {13., 19.}, PlotRange -> {{0., 13.}, {0., 19.}}];

	pauseStatic = Graphics[{Thickness[0.07692307692307693], FaceForm[{RGBColor[0.22, 0.502, 0.776], Opacity[1.]}], FilledCurve[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}}, {{{1.5, 1.5}, {4.5, 1.5}, {4.5, 17.5}, {1.5, 17.5}}}], FilledCurve[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}}, {{{8.5, 1.5}, {11.5, 1.5}, {11.5, 17.5}, {8.5, 17.5}}}]}, AspectRatio -> Automatic, ImageSize -> {13., 19.}, PlotRange -> {{0., 13.}, {0., 19.}}];

	cancelHover = Graphics[{Thickness[0.05263157894736842], FaceForm[{RGBColor[0.42700000000000005`, 0.718, 0.988], Opacity[1.]}], FilledCurve[{{{1, 4, 3}, {0, 1, 0}, {1, 3, 3}, {1, 3, 3}, {0, 1, 0}, {0, 1, 0}, {1, 3, 3}, {0, 1, 0}, {1, 3, 3}, {0, 1, 0}, {0, 1, 0}, {1, 3, 3}, {0, 1, 0}, {1, 3, 3}, {0, 1, 0}, {0, 1, 0}, {1, 3, 3}, {0, 1, 0}, {1, 3, 3}, {0, 1, 0}}, {{1, 4, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}}}, {CompressedData["
	1:eJxTTMoPSmViYGBQB2IQ7dyd8/x3prZDu9jNc98vSzk8/73y46Wz2g72pnG7
	PHkQfI+HVSLrjks4wNQzgIGEw38w0HSofKlmyKEjClWvCVUvAjGnR9NB/64K
	W+NUEYfpeULNB7wQ/BqQAk4E/0oF0KATGnD9e0omS7BM04CbD7FXGcpXcGiD
	uhsmD3O3IccamagUEQd3qLthfIh7xeHqK1DMlXAwgKqD+RfGh5kLUw8Lrwqo
	O2DugsnD3A3Tn/D0gtLtnwh+TP+hrxpzNNHs14K7zwUazjD344oXmP9dUOJF
	Geo+Vbg7YPIwd8D0w9wB48PiE6YeNb61cOqDmQtTD7O3DeoOmLsAF/ADSw==

	"], {{9.5, 17.5}, {5.081999999999999, 17.5}, {1.5, 13.918}, {1.5, 9.5}, {1.5, 5.082}, {5.081999999999999, 1.5}, {9.5, 1.5}, {13.918, 1.5}, {17.5, 5.082}, {17.5, 9.5}, {17.5, 13.918}, {13.918, 17.5}, {9.5, 17.5}}}]}, AspectRatio -> Automatic, ImageSize -> {19., 19.}, PlotRange -> {{0., 19.}, {0., 19.}}];

	cancelStatic = Graphics[{Thickness[0.05263157894736842], FaceForm[{RGBColor[0.537, 0.537, 0.537], Opacity[1.]}], FilledCurve[{{{1, 4, 3}, {0, 1, 0}, {1, 3, 3}, {1, 3, 3}, {0, 1, 0}, {0, 1, 0}, {1, 3, 3}, {0, 1, 0}, {1, 3, 3}, {0, 1, 0}, {0, 1, 0}, {1, 3, 3}, {0, 1, 0}, {1, 3, 3}, {0, 1, 0}, {0, 1, 0}, {1, 3, 3}, {0, 1, 0}, {1, 3, 3}, {0, 1, 0}}, {{1, 4, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}}}, {CompressedData["
	1:eJxTTMoPSmViYGBQB2IQ7dyd8/x3prZDu9jNc98vSzk8/73y46Wz2g72pnG7
	PHkQfI+HVSLrjks4wNQzgIGEw38w0HSofKlmyKEjClWvCVUvAjGnR9NB/64K
	W+NUEYfpeULNB7wQ/BqQAk4E/0oF0KATGnD9e0omS7BM04CbD7FXGcpXcGiD
	uhsmD3O3IccamagUEQd3qLthfIh7xeHqK1DMlXAwgKqD+RfGh5kLUw8Lrwqo
	O2DugsnD3A3Tn/D0gtLtnwh+TP+hrxpzNNHs14K7zwUazjD344oXmP9dUOJF
	Geo+Vbg7YPIwd8D0w9wB48PiE6YeNb61cOqDmQtTD7O3DeoOmLsAF/ADSw==

	"], {{9.5, 17.5}, {5.081999999999999, 17.5}, {1.5, 13.918}, {1.5, 9.5}, {1.5, 5.082}, {5.081999999999999, 1.5}, {9.5, 1.5}, {13.918, 1.5}, {17.5, 5.082}, {17.5, 9.5}, {17.5, 13.918}, {13.918, 17.5}, {9.5, 17.5}}}]}, AspectRatio -> Automatic, ImageSize -> {19., 19.}, PlotRange -> {{0., 19.}, {0., 19.}}];

	commonCancelStopStatic = 
	  Image[ImageData[cancelStatic] /. {_, _, _, _} -> {1, 1, 1, 1}];

	startHover = Graphics[{Thickness[0.047619047619047616`], FaceForm[{RGBColor[0.988, 0.502, 0.51], Opacity[1.]}], FilledCurve[{{{1, 4, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}}}, {{{10.5, 16.5}, {7.187, 16.5}, {4.5, 13.812999999999999`}, {4.5, 10.5}, {4.5, 7.186999999999999}, {7.187, 4.5}, {10.5, 4.5}, {13.812999999999999`, 4.5}, {16.5, 7.186999999999999}, {16.5, 10.5}, {16.5, 13.812999999999999`}, {13.812999999999999`, 16.5}, {10.5, 16.5}}}], FilledCurve[{{{1, 4, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}}, {{1, 4, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}}}, CompressedData["
	1:eJxTTMoPSmVmYGBgAmJeKA0Bqg4QmsWhXezmue/BEnA+fnGEvj0lkyVYrumi
	qmswQlUH5MPUocjjEkfSBzMXRR0Wd6DqM3ZgDuPT3SQrBudD5H/Y/7tS8VLt
	ox6cj2rOD3sWqD4U/VjUYTUHaA8LVnsR7oLpQ9GPRR0AsZRRDQ==
	"]]}, AspectRatio -> Automatic, ImageSize -> {21., 21.}, PlotRange -> {{0., 21.}, {0., 21.}}];

	startStatic = Graphics[{Thickness[0.047619047619047616`], FaceForm[{RGBColor[0.8, 0., 0.], Opacity[1.]}], FilledCurve[{{{1, 4, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}}}, {{{10.5, 16.5}, {7.187, 16.5}, {4.5, 13.812999999999999`}, {4.5, 10.5}, {4.5, 7.186999999999999}, {7.187, 4.5}, {10.5, 4.5}, {13.812999999999999`, 4.5}, {16.5, 7.186999999999999}, {16.5, 10.5}, {16.5, 13.812999999999999`}, {13.812999999999999`, 16.5}, {10.5, 16.5}}}], FilledCurve[{{{1, 4, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}}, {{1, 4, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}}}, CompressedData["
	1:eJxTTMoPSmVmYGBgAmJeKA0Bqg4QmsWhXezmue/BEnA+fnGEvj0lkyVYrumi
	qmswQlUH5MPUocjjEkfSBzMXRR0Wd6DqM3ZgDuPT3SQrBudD5H/Y/7tS8VLt
	ox6cj2rOD3sWqD4U/VjUYTUHaA8LVnsR7oLpQ9GPRR0AsZRRDQ==
	"]]}, AspectRatio -> Automatic, ImageSize -> {21., 21.}, PlotRange -> {{0., 21.}, {0., 21.}}];

	commonStartPauseStatic = 
	  Image[ImageData[startStatic] /. {_, _, _, _} -> {1., 1., 1., 1.}];

	graphWidth = 152.;

	elideExpand = 
	  Graphics[{Thickness[0.08333333333333333], 
	    FaceForm[{RGBColor[0.533, 0.533, 0.533], Opacity[1.]}], 
	    FilledCurve[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 
	        0}}}, 
	         {{{6., 1.58589999999999199}, {1.7930000000000001, 
	        5.7928999999999995}, {3.207, 7.206899999999999}, {6., 
	        4.4139}, {8.793, 7.206899999999999}, {10.207, 
	        5.7928999999999995}}}]}, 
	     AspectRatio -> Automatic, ImageSize -> {graphWidth, 9.}, 
	   PlotRange -> {{0., 12.}, {0., 9.}}];

	elideCollapse = 
	  Graphics[{Thickness[0.08333333333333333], 
	    FaceForm[{RGBColor[0.533, 0.533, 0.533], Opacity[1.]}], 
	    FilledCurve[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 
	        0}}}, {{{6., 7.4140999999999995}, {10.207, 
	        3.2070999999999996}, {8.793, 1.7931}, {6., 4.5861}, {3.207, 
	        1.7931}, {1.7930000000000001, 3.2070999999999996}}}]}, 
	   AspectRatio -> Automatic, ImageSize -> {graphWidth, 9.}, 
	   PlotRange -> {{0., 12.}, {0., 9.}}];

	(* END: Icon image assets *)

	(* START: Average label placeholder *)

	averageLabelPH = 
	  Rasterize[Style["0 000 000", 10, FontFamily -> "Roboto", White]];

	(* END: Average label placeholder *)

	(* START: Arrow transparency functions *)

	bcarrow[fac_] := Graphics[{Thickness[0.03333333333333333],
	       {RGBColor[0.169, 0.69, 0.8510000000000001], 
	     Thickness[0.06666666666666667], Opacity[fac*1.], 
	     CapForm["Round"], JoinForm["Round"], 
	         JoinedCurve[{{{0, 2, 0}, {0, 1, 0}}}, {{{12., 7.}, {7.5, 
	         3.}, {3., 7.}}}, CurveClosed -> {0}]}, {RGBColor[0.169, 
	      0.69, 0.8510000000000001], Thickness[0.06666666666666667], 
	         Opacity[fac*1.], CapForm["Round"], JoinForm["Round"], 
	     JoinedCurve[{{{0, 2, 0}}}, {{{7.5, 4.5}, {7.5, 12.5}}}, 
	      CurveClosed -> {0}]}}, AspectRatio -> Automatic, 
	   ImageSize -> {30., 16.}, 
	     PlotRange -> {{0., 30.}, {0., 16.}}];
	bgarrow[fac_] := Graphics[{Thickness[0.03333333333333333], 
	       {RGBColor[0.749, 0.749, 0.749], 
	     Thickness[0.06666666666666667], Opacity[fac*1.], 
	     CapForm["Round"], JoinForm["Round"], 
	         JoinedCurve[{{{0, 2, 0}, {0, 1, 0}}}, {{{12., 7.}, {7.5, 
	         3.}, {3., 7.}}}, CurveClosed -> {0}]}, {RGBColor[0.749, 
	      0.749, 0.749], Thickness[0.06666666666666667], Opacity[fac*1.], 
	         CapForm["Round"], JoinForm["Round"], 
	     JoinedCurve[{{{0, 2, 0}}}, {{{7.5, 4.5}, {7.5, 12.5}}}, 
	      CurveClosed -> {0}]}}, AspectRatio -> Automatic, 
	   ImageSize -> {30., 16.}, 
	     PlotRange -> {{0., 30.}, {0., 16.}}];

	rcarrow[fac_] := 
	  Graphics[{Thickness[
	     0.03333333333333333], {RGBColor[0.894, 0.196, 0.196], 
	     Thickness[0.06666666666666667], Opacity[fac*1.], 
	     CapForm["Round"], JoinForm["Round"], 
	         JoinedCurve[{{{0, 2, 0}, {0, 1, 0}}}, {{{27., 9.}, {22.5, 
	         13.}, {18., 9.}}}, CurveClosed -> {0}]}, {RGBColor[0.894, 
	      0.196, 0.196], Thickness[0.06666666666666667], Opacity[fac*1.], 
	         CapForm["Round"], JoinForm["Round"], 
	     JoinedCurve[{{{0, 2, 0}}}, {{{22.5, 11.5}, {22.5, 3.5}}}, 
	      CurveClosed -> {0}]}}, AspectRatio -> Automatic, 
	   ImageSize -> {30., 16.}, 
	     PlotRange -> {{0., 30.}, {0., 16.}}];
	rgarrow[fac_] := 
	  Graphics[{Thickness[
	     0.03333333333333333], {RGBColor[0.749, 0.749, 0.749], 
	     Thickness[0.06666666666666667], Opacity[fac*1.], 
	     CapForm["Round"], JoinForm["Round"], 
	         JoinedCurve[{{{0, 2, 0}, {0, 1, 0}}}, {{{27., 9.}, {22.5, 
	         13.}, {18., 9.}}}, CurveClosed -> {0}]}, {RGBColor[0.749, 
	      0.749, 0.749], Thickness[0.06666666666666667], Opacity[fac*1.], 
	         CapForm["Round"], JoinForm["Round"], 
	     JoinedCurve[{{{0, 2, 0}}}, {{{22.5, 11.5}, {22.5, 3.5}}}, 
	      CurveClosed -> {0}]}}, AspectRatio -> Automatic, 
	   ImageSize -> {30., 16.}, 
	     PlotRange -> {{0., 30.}, {0., 16.}}];

	carrow[fac_] := 
	  Graphics[{Thickness[
	     0.03333333333333333], {RGBColor[0.894, 0.196, 0.196], 
	     Thickness[0.06666666666666667], Opacity[fac*1.], 
	     CapForm["Round"], JoinForm["Round"], 
	         JoinedCurve[{{{0, 2, 0}, {0, 1, 0}}}, {{{27., 9.}, {22.5, 
	         13.}, {18., 9.}}}, CurveClosed -> {0}]}, {RGBColor[0.894, 
	      0.196, 0.196], Thickness[0.06666666666666667], Opacity[fac*1.], 
	         CapForm["Round"], JoinForm["Round"], 
	     JoinedCurve[{{{0, 2, 0}}}, {{{22.5, 11.5}, {22.5, 3.5}}}, 
	      CurveClosed -> {0}]}, 
	       {RGBColor[0.169, 0.69, 0.8510000000000001], 
	     Thickness[0.06666666666666667], Opacity[fac*1.], 
	     CapForm["Round"], JoinForm["Round"], 
	         JoinedCurve[{{{0, 2, 0}, {0, 1, 0}}}, {{{12., 7.}, {7.5, 
	         3.}, {3., 7.}}}, CurveClosed -> {0}]}, {RGBColor[0.169, 
	      0.69, 0.8510000000000001], Thickness[0.06666666666666667], 
	         Opacity[fac*1.], CapForm["Round"], JoinForm["Round"], 
	     JoinedCurve[{{{0, 2, 0}}}, {{{7.5, 4.5}, {7.5, 12.5}}}, 
	      CurveClosed -> {0}]}}, AspectRatio -> Automatic, 
	   ImageSize -> {30., 16.}, 
	     PlotRange -> {{0., 30.}, {0., 16.}}];
	garrow[fac_] := 
	  Graphics[{Thickness[
	     0.03333333333333333], {RGBColor[0.749, 0.749, 0.749], 
	     Thickness[0.06666666666666667], Opacity[fac*1.], 
	     CapForm["Round"], JoinForm["Round"], 
	         JoinedCurve[{{{0, 2, 0}, {0, 1, 0}}}, {{{27., 9.}, {22.5, 
	         13.}, {18., 9.}}}, CurveClosed -> {0}]}, {RGBColor[0.749, 
	      0.749, 0.749], Thickness[0.06666666666666667], Opacity[fac*1.], 
	         CapForm["Round"], JoinForm["Round"], 
	     JoinedCurve[{{{0, 2, 0}}}, {{{22.5, 11.5}, {22.5, 3.5}}}, 
	      CurveClosed -> {0}]}, 
	       {RGBColor[0.749, 0.749, 0.749], 
	     Thickness[0.06666666666666667], Opacity[fac*1.], 
	     CapForm["Round"], JoinForm["Round"], 
	         JoinedCurve[{{{0, 2, 0}, {0, 1, 0}}}, {{{12., 7.}, {7.5, 
	         3.}, {3., 7.}}}, CurveClosed -> {0}]}, {RGBColor[0.749, 
	      0.749, 0.749], Thickness[0.06666666666666667], Opacity[fac*1.], 
	         CapForm["Round"], JoinForm["Round"], 
	     JoinedCurve[{{{0, 2, 0}}}, {{{7.5, 4.5}, {7.5, 12.5}}}, 
	      CurveClosed -> {0}]}}, AspectRatio -> Automatic, 
	   ImageSize -> {30., 16.}, 
	     PlotRange -> {{0., 30.}, {0., 16.}}];

	(* END: Arrow transparency functions *)

	(* START: Arrow lists *)

	inListDimStage = 
	  Table[Overlay[{rgarrow[i], rcarrow[1 - i]}], {i, 0, 0.5, 0.1}];
	inList = Join[
	   inListDimStage,
	   Reverse[inListDimStage]];

	outListDimStage = 
	  Table[Overlay[{bgarrow[i], bcarrow[1 - i]}], {i, 0, 0.5, 0.1}];
	outList = Join[
	   outListDimStage,
	   Reverse[outListDimStage]];

	toPacketsList = 
	  Table[Overlay[{carrow[i], garrow[1 - i]}], {i, 0, 1, 0.1}];
	PrependTo[toPacketsList, toPacketsList[[1]]];

	(* END: Arrow lists *)

	(* START: Create a formatted time object from a given Unix time *)

	buildTime[timePassed_Integer] :=
	  Module[{min, sec},
	   sec = Mod[timePassed, 60];
	   min = Floor[timePassed/60];
	   Style[
	    StringJoin[StringPadLeft[ToString[min], 2, "0"], ":", 
	     StringPadLeft[ToString[sec], 2, "0"]],
	    10, FontFamily -> "Roboto", RGBColor[0.2196, 0.50195, 0.77255]
	    ]
	   ];

	(* END: Create a formatted time object from a given Unix time *)

	(* START: Textual labels *)

	averagePackets = 
	  Style["pps", 10, FontFamily -> "Roboto", 
	   RGBColor[0.39215, 0.39215, 0.39215]];

	timeInitial = 
	  Style["00:00", 10, FontFamily -> "Roboto", 
	   RGBColor[0.749019, 0.749019, 0.749019]];

	(* END: Textual labels *)

    (* START: Interface *)

    $interfaceToUse := $NetworkInterfaces;

    (* END: Interface*)

	(* START: NetworkPacketCapture[] *)

	System`NetworkPacketCapture[filterSpec_?AssociationQ] := 
		Block[
	        {
	            times = {},
	            shouldContinue = True,
	            notPaused = True,
	            captures = {},
	            chartLabelPH,
	            resDataset = $Failed,
	            captureOn = False,

	            averageLabel,
	            packetsOffOn,
	            timeInitialOngoing,
	            graph,
	            chartLabel,
	            refTime,
	            
	            initial,
	            recording,
	            expand,
	            
	            overallRefRate,
	            toPacketsOnRR,
	            offset,
	            dimPacketsInRR,
	            inOutRRDifference,
	            dimPacketsOutRR,
	            
	            packetSpeed,
	            
	            cancelStopStatic,
	            
	            toGraph,
	            toNoGraph,
	            
	            speeds,
	            maxPacketSpeed,
	            pause,
	            
	            startPauseStatic,

	            tempCell,

	            interface,

	            fullStopCapture,

	            sPSHovers,

	            cSSHovers,

	            startTime,
	            timeSoFar,

	            havePackets = False,

		    	lastPacketSpeedTime = UnixTime[],

		    	packetNumOffset, intm
	        },	

	        If[!Developer`UseFrontEnd[CurrentValue["UserInteractionEnabled"]],
				Message[NetworkPacketCapture::fereq];
				Return[$Failed]
			];
			interface = determineInterface[
				Merge[
					{
						filterSpec,
						Association["Interface" -> $interfaceToUse]
					},
					First
				]
			];

			If[!testInterface[],
				Switch[$OperatingSystem,
					"MacOSX", Message[System`NetworkPacketCapture::permosx],
					"Windows", Message[System`NetworkPacketCapture::permwin],
					_, Message[System`NetworkPacketCapture::permlinux]
				];
				Return[$Failed]
			];

			fullStopCapture[] := Module[{},
		    	stopCapture[interface];
		    	fixDataset[getFullCapture[]]
		    ];

	        pause[] := Module[{},
				timeInitialOngoing = buildTime[Abs[Total[Subtract @@ #1 & /@ times]]];
				
				startPauseStatic = PaneSelector[{False -> #1, True -> #2}, Dynamic[(
					sPSHovers = Append[
							sPSHovers[[-10 ;;]],
							CurrentValue["MouseOver"]
						];
					MemberQ[sPSHovers, True])]]& @@ (
					Button[
						Overlay[{commonStartPauseStatic, #1}],
						recording[],
						Appearance -> None,
						ImageSize -> {21, 21}
					] & /@ {startStatic, startHover});
				packetsOffOn = Overlay[{rgarrow[0], rcarrow[1], bgarrow[0], bcarrow[1]}];
				averageLabel = Overlay[{
						averageLabelPH,
						Style[ToString[0], 10, FontFamily -> "Roboto", RGBColor[0.39215, 0.39215, 0.39215]]
					},
					Alignment -> Right
				];	                 
				notPaused = False;
	            AppendTo[captures, fullStopCapture[]];
	        ];

	        cancel[] := Module[{},

				notPaused = False;

				shouldContinue = False;

				havePackets = True;

				NotebookDelete[tempCell];

	        ];

	        recording[] :=
	          Module[{triesi, triesj},
	            notPaused = True;
	           
	            For[triesi = 0, triesi < 1000, triesi++,

		            For[triesj = 0, triesj < 1000 && (
		            		(StringQ[interface] && !MemberQ[iGetActivePacketCaptures[], interface]) ||
		            		(ListQ[interface] && !SubsetQ[iGetActivePacketCaptures[], interface]) ||
		            		(interface === All && !SubsetQ[iGetActivePacketCaptures[], $NetworkInterfaces])
		            	),  triesj++,
		                startCapture[interface, Association[]];
		            ];

		            If[triesj >= 1000,
		                Continue[];
		            ];
		           
		        	For[triesj = 0, triesj < 100 && (tmpPS = GetPacketSpeed[interface])[[1]] <= 0 && tmpPS[[2]] <= 0,  triesj++, Null];

		            If[triesj >= 100,
		                Continue[];
		            ];

		            Break[];

		        ];

	        	If[triesi >= 1000,
	                resDataset = $Failed;
	                shouldContinue = False;
	                Return[];
	            ];


	            startTime = UnixTime[];

	            captureOn = True;

	            timeSoFar = Abs[Total[Subtract @@ #1 & /@ times]];
	           
	            timeInitialOngoing = Hold[Refresh[buildTime[timeSoFar + (UnixTime[] - startTime)], UpdateInterval -> 1]];
	           
	            overallRefRate = 1/4;

	            toPacketsOnRR = 1/2;

	            

	            offset = 0;
	            dimPacketsInRR = (2)/Length[inList];
	            inOutRRDifference = 1;
	            dimPacketsOutRR := dimPacketsInRR*inOutRRDifference + offset;

	            
	           
	            packetsOffOn = Hold[Refresh[
	                If[
	                    Round[AbsoluteTime[]/toPacketsOnRR - refTime/toPacketsOnRR] < Length[toPacketsList],
	                    toPacketsList[[(Round[AbsoluteTime[]/toPacketsOnRR - refTime/toPacketsOnRR]) + 1]]
	                    ,
	                    Overlay[
	                        {
	                            inList[[
	                                Mod[
	                                    Round[AbsoluteTime[]/dimPacketsInRR - refTime/dimPacketsInRR],
	                                    Length[inList],
	                                    1
	                                ]
	                            ]],
	                            outList[[
	                                Mod[
	                                    Round[AbsoluteTime[]/dimPacketsOutRR - refTime/dimPacketsOutRR],
	                                    Length[outList],
	                                    1
	                                ]
	                            ]]
	                        }
	                    ]
	                ]
	                ,
	                UpdateInterval -> overallRefRate
	            ]];

				averageLabel =
					Hold[Refresh[
						If[UnixTime[] != lastPacketSpeedTime,
							packetSpeed = GetPacketSpeed[interface];
							lastPacketSpeedTime = UnixTime[];
						];

        					If[packetSpeed != speeds[[-1]] && notPaused,
							speeds = Append[
								speeds[[-10 ;;]],
								packetSpeed
							];
						];


						If[packetSpeed[[2]] <= 0,
							offset = (2)/Length[outList];
							dimPacketsInRR = Infinity;
							If[packetSpeed[[1]] <= 0,
								packetSpeed = {0, 0};
								offset = Infinity;
							];
							,
							offset = 0;
							dimPacketsInRR = (2)/Length[inList];
							inOutRRDifference = 1;
							dimPacketsOutRR := dimPacketsInRR*inOutRRDifference + offset;
							inOutRRDifference = packetSpeed[[1]]/packetSpeed[[2]];
							If[inOutRRDifference == 0,
								offset = Infinity;
							];
						];

						Overlay[{
							averageLabelPH,
							Style[ToString[Total[packetSpeed]], 10, FontFamily -> "Roboto", RGBColor[0.39215, 0.39215, 0.39215]]
							},
							Alignment -> Right
						]
						,
						UpdateInterval -> 1
					]
				];

				chartLabel = Sequence[Style["\[DownArrow] In ", 10, FontFamily -> "Roboto", blueChart], Style["\[UpArrow] Out", 10, FontFamily -> "Roboto", redChart]];

				startPauseStatic =
					(PaneSelector[
						{False -> #1, True -> #2},
						Dynamic[(
							sPSHovers = Append[
								sPSHovers[[-10 ;;]],
								CurrentValue["MouseOver"]
							];
							MemberQ[sPSHovers, True]
						)]
					]& @@ (Button[
						Overlay[{commonStartPauseStatic, #1}],

						AppendTo[times, {startTime, UnixTime[]}];
						pause[];
						Return[];,

						Appearance -> None,
						ImageSize -> {21, 21}
					] & /@ {pauseStatic, pauseHover}));

				cancelStopStatic =
					(PaneSelector[
						{False -> #1, True -> #2},
						Dynamic[(cSSHovers = Append[
							cSSHovers[[-10 ;;]],
							CurrentValue["MouseOver"]
						];
						MemberQ[cSSHovers, True])]
					]& @@ (Button[
	                    Overlay[{commonCancelStopStatic, #1}, Alignment -> Right],
	                     
	                    AppendTo[times, {startTime, UnixTime[]}]; 
	                    cancel[];
	                    Return[];,
	                     
	                    Appearance -> None,
	                    ImageSize -> ImageDimensions[commonCancelStopStatic]
	                ] & /@ {stopStatic, stopHover}));
	        ];

	        toNoGraph[] := Module[{},
				chartLabelPH = Hold[Sequence[]];
				graph = Hold[Sequence[]];
				expand = Item[EventHandler[elideExpand, {"MouseClicked" :> toGraph[]}], Background -> RGBColor[0.898039, 0.898039, 0.898039],
						Alignment -> Bottom
					];
			];

			toGraph[] := Module[{},
				chartLabelPH = Hold[{Row[{chartLabel}, BaselinePosition -> Scaled[.6]]}];
				maxPacketSpeed = 100;
				graph = {
							Hold[Refresh[
								maxPacketSpeed = Max[maxPacketSpeed, speeds];
								Graphics[
									{
										If[!captureOn,
											Opacity[0]
										],
										Table[
											{
												If[
													speeds[[i]] == {-1, -1},
													Opacity[0]
												],
												blueChart, 
												Line[{{i, 0}, {i, speeds[[i, 1]]}}],
												redChart,
												Line[{{i + 0.2, 0}, {i + 0.2, speeds[[i, 2]]}}]
											}, {i, 1, 10}]
									},
									AspectRatio -> GoldenRatio^(-1), Axes -> {False, False}, 
									AxesLabel -> {None, None}, 
									AxesOrigin -> {-0.19272727272727308, 0.}, 
									Background -> RGBColor[0.960784, 0.960784, 0.960784], 
									CoordinatesToolOptions :> {"DisplayFunction" -> ({Identity[#1[[1]]], Identity[#1[[2]]]} & ), 
									"CopiedValueFunction" -> ({Identity[#1[[1]]], Identity[#1[[2]]]} & )}, 
									FrameLabel -> {{None, None}, {None, None}}, 
									FrameTicks -> {Automatic, Automatic}, 
									GridLines -> {None, Range[0, (6*maxPacketSpeed)/5, (1*maxPacketSpeed)/5]}, 
									GridLinesStyle -> Directive[GrayLevel[0.5, 0.4]], 
									ImageSize -> {graphWidth, Automatic},
									PlotRange -> {{All, All}, {0, (6*maxPacketSpeed)/5}}, 
									PlotRangePadding -> {{Scaled[0.02], Scaled[0.02]}, {Scaled[0.00], Scaled[0.00]}}, 
									Ticks -> {None, Automatic}
								],
								UpdateInterval -> 1
							]], 
							SpanFromLeft
						};
				expand = Item[EventHandler[elideCollapse, {"MouseClicked" :> toNoGraph[]}], Background -> RGBColor[0.898039, 0.898039, 0.898039],
						Alignment -> Bottom
					];
			];

	        (* START: Initial *)

        	speeds = Table[{-1,-1},{11}];
        	
            chartLabelPH = Hold[Sequence[]];

            averageLabel = 
                Overlay[
                    {
                    averageLabelPH,
                    Style["0",10,FontFamily->"Roboto",RGBColor[0.749019,0.749019,0.749019]]
                    },
                    Alignment->Right
                ];

            packetsOffOn = 
                Overlay[{rgarrow[1], rcarrow[0], bgarrow[1], bcarrow[0]}];

            timeInitialOngoing = timeInitial;

            graph = Hold[Sequence[]];

            chartLabel = Sequence[Style["\[DownArrow] In ", 10, FontFamily -> "Roboto", RGBColor[0.749, 0.749, 0.749]], Style["\[UpArrow] Out", 10, FontFamily -> "Roboto", RGBColor[0.749, 0.749, 0.749]]];

            expand = Item[EventHandler[elideExpand,
                    {"MouseClicked" :> toGraph[]}
                ], Background -> RGBColor[0.898039, 0.898039, 0.898039],
                	Alignment -> Bottom
                ];

            sPSHovers = Table[False, {10}];

           	startPauseStatic = PaneSelector[{False -> #1, True -> #2}, Dynamic[(sPSHovers = Append[
			sPSHovers[[-10 ;;]],
			CurrentValue["MouseOver"]
		];MemberQ[sPSHovers, True])]]& @@ (Button[
                    Overlay[{commonStartPauseStatic, #1}],
                    refTime = AbsoluteTime[];
                    recording[];
                    Return[];,
                    Appearance -> None,
                    ImageSize -> {21, 21}
                ] & /@ {startStatic, startHover});

            cSSHovers = Table[False, {10}];

            cancelStopStatic = PaneSelector[{False -> #1, True -> #2}, Dynamic[(cSSHovers = Append[
			cSSHovers[[-10 ;;]],
			CurrentValue["MouseOver"]
		];MemberQ[cSSHovers, True])]]& @@ (Button[
                Overlay[{commonCancelStopStatic, #1}, Alignment -> Right],
                resDataset = $Canceled;
                shouldContinue = False;
                Return[];,
                Appearance -> None,
                ImageSize -> ImageDimensions[commonCancelStopStatic]
            ] & /@ {cancelStatic, cancelHover});

	    	(* END: Initial *)

	        tempCell = PrintTemporary[
	            Dynamic[Framed[Grid[
	                {
	                    {Row[{startPauseStatic,ReleaseHold[packetsOffOn]},Spacer[{3,0}],BaselinePosition->Bottom],cancelStopStatic},
	                    {Row[{ReleaseHold[timeInitialOngoing]}],Row[{ReleaseHold[averageLabel],averagePackets}," "]},
	                    ReleaseHold[chartLabelPH]
	                    ,
	                    ReleaseHold[graph],
	                    {expand, SpanFromLeft}
	                },
	                Alignment->{Center,Automatic,
	                        {{1,1}->Left,{1,2}->{Right,Center},{2,1}->{Left},{2,2}->Right,{3,1}->Left,{3,2}->Right}
	                    },
	                BaselinePosition->{1,1},
	                Spacings->{{1,{},1},{.8,{.1},.5}},
	                ItemSize->{Full, Full},
	                Frame->All,
	                Editable->False,
			Selectable->False,
			FrameStyle->Directive[1,RGBColor[0,0,0,0]]],
	                FrameMargins->{{-1,-1},{-3,-2}},
	                ImageSize->{167.,Full},
	                FrameStyle->Directive[1,RGBColor[0.8196,0.8196,0.8196]],
	                Editable->False,
			Selectable->False,
			RoundingRadius->3], TrackedSymbols -> Full]
	        ];

	        While[shouldContinue];

	        If[havePackets,
	        	packetNumOffset = 0;
                resDataset = Join[
                    fullStopCapture[],
                    Sequence @@ Table[
                        intm = Replace[#1, Verbatim[Rule]["PacketNumber", x_Integer] -> x + packetNumOffset] & /@ captures[[i]];
                        packetNumOffset += Last[captures[[i]]]["PacketNumber"];
                        intm
                        ,
                        {i, 1, Length[captures]}
                    ]
                ];
	        ];

	        resDataset

	    ];

	    System`NetworkPacketCapture[
		    	iface_ /; Switch[
		    			iface,
		    			_String,
		    			Quiet[FailureQ[Interpreter["NetworkService"][iface]]],
		    			_List,
		    			AllTrue[
		    				iface,
		    				StringQ
		    			],
		    			All,
		    			True,
		    			_,
		    			False
		    		]
		    	, 
		    	args___
	    	] := 
	    	Block[
	    		{$inNPC = True, $interfaceToUse = iface}, 
	    		NetworkPacketCapture[args]
	    	] /; !TrueQ[$inNPC];

	    System`NetworkPacketCapture[] := System`NetworkPacketCapture[<||>];
		System`NetworkPacketCapture[port_?validPortFilterSpec] := System`NetworkPacketCapture[<|"Port"->port|>];
		System`NetworkPacketCapture[service_?StringQ] := Block[
		  { serviceEntity = Interpreter["NetworkService"][service] },
		  If[FailureQ[serviceEntity],
		    (Message[System`NetworkPacketCapture::noserv,service]; $Failed),
		    (*ELSE*)
		    (*got the entity, go down the recursion*)
		    System`NetworkPacketCapture[serviceEntity]
		  ]
		];

		System`NetworkPacketCapture[serviceEntity_Entity] := Block[
		  {},
		  If[TrueQ[EntityTypeName[serviceEntity] === "NetworkService"],
		    (
		      ports = serviceEntity["DefaultPorts"];
		      If[!MissingQ[ports] && !FailureQ[ports] && DeleteMissing[Union[Flatten@Values[ports]]] =!= {},
		        System`NetworkPacketCapture[<|"Port"->DeleteMissing[Union[Flatten@Values[ports]]]|>],
		        (
		          Message[System`NetworkPacketCapture::noport,serviceEntity];
		          System`NetworkPacketCapture[<||>]
		        )
		      ]
		    ),
		    (Message[System`NetworkPacketCapture::invalidentity,serviceEntity]; $Failed)
		  ]
		];

		(*rule form gets turned into an association*)
		System`NetworkPacketCapture[r_Rule] := System`NetworkPacketCapture[<|r|>];

	(* END: NetworkPacketCapture[] *)


]; (*End of disable in cloud. *)

$interfaceAvailable = False;
testInterface[] := If[$interfaceAvailable, True, If[iTestInterface[]>0, $interfaceAvailable=True, False]]

checkInteger[n_?IntegerQ] /; n > 0 := ToString[n]
checkInteger[___] := (Message[System`NetworkPacketTrace::portinvalid]; $Failed)

checkIntegerList[ns : {(_?IntegerQ) ..}] /; AllTrue[ns, # > 0 &] := StringRiffle[ToString /@ ns, " or port "]
checkIntegerList[___] := (Message[System`NetworkPacketTrace::portinvalid]; $Failed)

checkIPAddress[IPAddress[s_?StringQ]] := checkIPAddress[s]
checkIPAddress[s_?StringQ] /; Socket`IPv4AddressQ[s] || Socket`IPv6AddressQ[s] := s
checkIPAddress[___] := (Message[System`NetworkPacketTrace::ipaddrinvalid]; $Failed)

check2IntegerList[{n1_?IntegerQ, n2_?IntegerQ}] := StringRiffle[{n1, n2}, "-"]
check2IntegerList[Span[n1_?IntegerQ, n2_?IntegerQ]] /; n1 <= n2 := check2IntegerList[{n1, n2}]
check2IntegerList[Interval[{n1_?IntegerQ, n2_?IntegerQ}]] /; n1 <= n2 := check2IntegerList[{n1, n2}]
check2IntegerList[___] := (Message[System`NetworkPacketTrace::portinvalid]; $Failed)

checkStringValues[s_?StringQ, vals_] /; StringMatchQ[s, Alternatives @@ Last[vals], IgnoreCase -> True] := ToLowerCase[s]
checkStringValues[___,vals_] := (Message[System`NetworkPacketTrace::strinvalid,First[vals],Last[vals]]; $Failed)

$supportedProtocols = {"icmp", "udp", "tcp"};

keySpecs = <|
  "PortNumber" -> {"port", checkInteger}, 
  "PortList" -> {"port", checkIntegerList}, 
  "PortRange" -> {"portrange", check2IntegerList}, 
  "IPAddress" -> {"net", checkIPAddress},
  (*the ip address type doesn't actually have a "key",so just return empty string and it'll resolve correctly*)
  "IPAddressType" -> {"", checkStringValues[#, "IPAddressType"->{"ipv4", "ipv6"}] &}, 
  "Protocol" -> {"", checkStringValues[#, "Protocol"->$supportedProtocols] &}
|>;

keyNameValueFix[key_, value_] := key

keyNameValueFix["Port", Span[_Integer, _Integer]] := "PortRange"
keyNameValueFix["Port", 
  Interval[{_Integer, _Integer}]] := "PortRange"
keyNameValueFix["Port", {(_?IntegerQ) ...}] := "PortList"
keyNameValueFix["Port", _?IntegerQ] := "PortNumber"
makeFilterString[assoc_?AssociationQ] := 
  Block[{str, func, val}, 
   StringRiffle[
    StringTrim /@ KeyValueMap[
     Function[{key, value}, {str, func} = 
       Which[! StringQ[key], Return[$Failed, Block], 
        StringMatchQ[key, 
         StartOfString ~~ "Destination" ~~ __], (MapAt["dst " <> # &, 
          Lookup[keySpecs, 
           keyNameValueFix[
            StringDelete[key, StartOfString ~~ "Destination"], value],
            Return[$Failed, Block]], 1]), 
        StringMatchQ[key, 
         StartOfString ~~ "Source" ~~ __], (MapAt["src " <> # &, 
          Lookup[keySpecs, 
           keyNameValueFix[
            StringDelete[key, StartOfString ~~ "Source"], value], 
           Return[$Failed, Block]], 1]), 
        True, (Lookup[keySpecs, keyNameValueFix[key, value], 
          Return[$Failed, Block]])];
      val = func[value];
      If[! StringQ[val], Return[$Failed, Block]];
      str <> " " <> val],
      assoc
    ],
    {"(", ") and (", ")"}
  ]
]


fixDataset[ds_]:=Quiet[Dataset[Select[AssociationQ]@(Join[KeyDrop["Info"]@ #, KeyMap[# <> "IPAddress" &]@ #["Info"]["IP"], 
    AssociationThread[{"SourcePort", "DestinationPort"}, 
     List @@ #["Info"][[3]]["Ports"]],KeyTake[{"Info"}]@#] & /@ Normal[ds])]]

validPortFilterSpec = (
  (*integer greater than 0*)
  (IntegerQ[#] && # > 0) || 
  (*list of integers greater than 0*)
  MatchQ[#,{_?(IntegerQ[#] && #>0&)..}] ||
  (*Span[] or Interval[] of 2 increasing integers greater than 0*)
  MatchQ[#,
    Span[n1_?IntegerQ,n2_?IntegerQ] | 
    Interval[{n1_?IntegerQ,n2_?IntegerQ}
  ] /; (n1 < n2 && n1 > 0 && n2 > 0) ] &
)

(*all the valid filter specs currently supported*)
validFilterSpec = (
  (*normal association version*)
  AssociationQ[#] || 
  (*network service versions string or an entity*)
  StringQ[#] || 
  MemberQ[{Entity,Rule},Head[#]] ||
  (*port version - integer, list of integers, or Span[] or Interval[]*)
  validPortFilterSpec[#]&
  
)

If[TrueQ[$CloudEvaluation] && !TrueQ[Lookup[CloudSystem`KernelInitialize`$ConfigurationProperties, "AllowNetworkPacketFunctionality"]],
(* Running in cloud environment, define dummy functions that tell the user this functionality is not yet available. *)
	System`NetworkPacketRecording[___] := (Message[General::cloudf, HoldForm@NetworkPacketRecording]; $Failed);
	System`NetworkPacketTrace[___] := (Message[General::cloudf, HoldForm@NetworkPacketTrace]; $Failed);
	System`NetworkPacketRecordingDuring[___] := (Message[General::cloudf, HoldForm@NetworkPacketRecordingDuring]; $Failed);
,
(* Else define as usual *)

	(*number case becomes Quantity[t,"Seconds"]*)
	System`NetworkPacketRecording[t_?NumberQ] /; t>=0 := System`NetworkPacketRecording[t,<||>];
	System`NetworkPacketRecording[t_?NumberQ,spec_?validFilterSpec] /; t>=0 := System`NetworkPacketRecording[Quantity[t,"Seconds"],spec];

	(*unit case becomes NetworkPacketTrace*)
	System`NetworkPacketRecording[t_?QuantityQ] /; (CompatibleUnitQ[t, "Seconds"] && t>= Quantity[0,"Seconds"]) := 
		System`NetworkPacketRecording[t,<||>];
	System`NetworkPacketRecording[t_?QuantityQ,spec_?validFilterSpec] /; (CompatibleUnitQ[t, "Seconds"] && t>= Quantity[0,"Seconds"]) := 
		If[#===$Aborted || FailureQ[#], #, First[#]]& @ 
		System`NetworkPacketTrace[Pause[QuantityMagnitude@UnitConvert[t, "Seconds"]],spec];


	SetAttributes[System`NetworkPacketRecordingDuring,HoldFirst];
	System`NetworkPacketRecordingDuring[all___] := System`NetworkPacketTrace[all];

	SetAttributes[System`NetworkPacketTrace,HoldFirst];

	(*port number versions*)
	System`NetworkPacketTrace[expr_,port_?validPortFilterSpec] := 
		System`NetworkPacketTrace[expr,<|"Port"->port|>];

	(*string form used with NetworkService entities*)
	System`NetworkPacketTrace[expr_, service_?StringQ] := Block[
		{ serviceEntity = Interpreter["NetworkService"][service] },
		If[FailureQ[serviceEntity],
			(Message[System`NetworkPacketTrace::noserv,service]; $Failed),
		(*ELSE*)
			(*got the entity, go down the recursion*)
			System`NetworkPacketTrace[expr,serviceEntity]
		]
	];

	System`NetworkPacketTrace[expr_,serviceEntity_Entity] := Block[{},
		If[TrueQ[EntityTypeName[serviceEntity] === "NetworkService"],
		    (*THEN*)
		    (*got the entity, look up the port numbers*)
		    (
		      ports = serviceEntity["DefaultPorts"];
		      If[!MissingQ[ports] && !FailureQ[ports] && DeleteMissing[Union[Flatten@Values[ports]]] =!= {},
		        (*THEN*)
		        (*then we have ports we can join together and use*)
		        System`NetworkPacketTrace[expr,<|"Port"->DeleteMissing[Union[Flatten@Values[ports]]]|>],
		        (*ELSE*)
		        (*no ports were found, so just continue with no filter*)
		        (
		          Message[System`NetworkPacketTrace::noport,serviceEntity];
		          System`NetworkPacketTrace[expr]
		        )
		      ]
		    ),
		    (*ELSE*)
		    (*wrong entity type or invalid entity*)
		    (Message[System`NetworkPacketTrace::invalidentity,serviceEntity]; $Failed)
		]
	];

	(*rule form gets turned into an association*)
	System`NetworkPacketTrace[expr_,r_Rule] := System`NetworkPacketTrace[expr,<|r|>];


	(*association form*)
	System`NetworkPacketTrace[expr_] := System`NetworkPacketTrace[expr,<||>];
	System`NetworkPacketTrace[expr_, filters_?AssociationQ] := Block[
	  {
	    iface,
	    packets,
	    abortQ = False,
	    exprResult
	  },
	  (
	    (*first check if the interface key exists in the association*)
	    If[KeyExistsQ[filters,"Interface"],
	      Which[
	        (*single interface contained in $NetworkInterface*)
	        MemberQ[$NetworkInterfaces,filters["Interface"]],
	        iface = {filters["Interface"]},
	        (*subset of interfaces in $NetworkInterfaces*)
	        ListQ[filters["Interface"]] && AllTrue[filters["Interface"],StringQ] && SubsetQ[$NetworkInterfaces,filters["Interface"]],
	        iface = filters["Interface"],
	        (*error*)
	        True,
	        (
	          Message[System`NetworkPacketTrace::iface,Short[filters["Interface"]]];
	          Return[$Failed]
	        )
	      ],
	      (*ELSE*)
	      (*no "Interface" key specified, default to all of $NetworkInterfaces*)
	      iface = System`$NetworkInterfaces
	    ];

		If[!testInterface[],
			Switch[$OperatingSystem,
				"MacOSX", Message[System`NetworkPacketTrace::permosx],
				"Windows", Message[System`NetworkPacketTrace::permwin],
				_, Message[System`NetworkPacketTrace::permlinux]
			];
			Return[$Failed]
		];

	    (*now check on the filter spec*)
	    Which[
	      KeyExistsQ[filters,"PCAPFilter"],
	      If[StringQ[filters["PCAPFilter"]],
	        (*THEN*)
	        (*it's a string*)
	        filterStringSpec = filters["PCAPFilter"],
	        (*ELSE*)
	        (*error - the pcap filter string must be a string*)
	        (
	          Message[System`NetworkPacketTrace::pcapfilterstring,Short[filters["PCAPFilter"]]];
	          Return[$Failed]
	        )
	      ],
	      (*ELSE*)
	      (*no filter string and the association isn't empty*)
	      Keys[filters] =!= {},
	      (
	        filterStringSpec = makeFilterString[KeyDrop[{"PCAPFilter","Interface"}]@filters]/.{"()" -> ""};
	        If[FailureQ[filterStringSpec], Return[$Failed]]
	      ),
	      True,
	      filterStringSpec=""
	    ];

	    (*start the capture on the specified interface setting with the filter string*)
	    (*we need to protect the abort*)
	    AbortProtect[
	      abortQ = CheckAbort[
	        (
	          StartPacketCapture[iface, filterStringSpec]; 
	          exprResult = expr;
	          False
	        ),
	        True
	      ]; 
	      packets = fixDataset@StopPacketCapture[]
	    ];
	    If[abortQ,$Aborted,{packets,exprResult}]
	  )
	];

]; (*End disable in cloud. *)


StartPacketCapture[] := StartPacketCapture[$DefaultNetworkInterface];
StartPacketCapture[iface_String] := iStartPacketCapture[iface, ""];
StartPacketCapture[iface_String, filter_String] := iStartPacketCapture[iface, filter];
StartPacketCapture[ifaces_List] := StartPacketCapture[ifaces, ""];
StartPacketCapture[ifaces_List, filter_String] := Scan[StartPacketCapture[#, filter] &, ifaces];

GetActivePacketCaptures[] := iGetActivePacketCaptures[];

GetPacketSpeed[iface_String] := iGetPacketSpeed[iface];
GetPacketSpeed[ifaces_List] := Total[GetPacketSpeed /@ ifaces];

StopPacketCapture[] := DatasetFrom[iStopAllPacketCaptures[]];
StopPacketCapture[ifaces_List] := DatasetFrom[Apply[iStopPacketCapture][ifaces]];
StopPacketCapture[iface_String] := StopPacketCapture[{iface}];

ImportPacketCapture[File[path_String],opts:OptionsPattern[]] := ImportPacketCapture[path];

ImportPacketCapture[path_String,opts:OptionsPattern[]] := 
	If[path === "" || !FileExistsQ[path],
		(*THEN*)
		(*the file doesn't exist*)
		$Failed,
		(*ELSE*)
		(*the file exists and we can continue - note that we should expand the filename first, as it only works with absolute file names*)
		{"Data"->fixDataset@DatasetFileFrom[iImportPacketCapture[ExpandFileName@path]]}
	];


If[TrueQ[$CloudEvaluation] && !TrueQ[Lookup[CloudSystem`KernelInitialize`$ConfigurationProperties, "AllowNetworkPacketFunctionality"]],
(* Running in cloud environment, define dummy functions that tell the user this functionality is not yet available. *)
	System`$DefaultNetworkInterface := (Message[General::cloudf, HoldForm@$DefaultNetworkInterface]; $Failed);
	System`$NetworkInterfaces := (Message[General::cloudf, HoldForm@$NetworkInterfaces]; $Failed);
,
(* Else define as usual *)
	System`$DefaultNetworkInterface := iGetDefaultInterface[];
	System`$NetworkInterfaces := iGetAllInterfaces[];
]; (* End disable in cloud. *)

(*reprotect all the System` symbols again*)

SetAttributes[#,{ReadProtected,Protected}]&/@
  {
    "System`NetworkPacketRecording",
    "System`NetworkPacketTrace",
    "System`NetworkPacketCapture",
    "System`$NetworkInterfaces",
    "System`$DefaultNetworkInterface"
  }

End[]

EndPackage[]

