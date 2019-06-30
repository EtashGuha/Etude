BeginPackage["TINSLink`ImportExport`"]

TINSLink`ImportExport`Symbols`ImportPacketCapture::usage = "imports a pcap file";

Begin["`Private`"]


Needs["TINSLink`"];

TINSLink`ImportExport`Symbols`ImportPacketCapture[path_String,opts:OptionsPattern[]] := Block[{res},
	If[path === "" || !FileExistsQ[path],
		(*THEN*)
		(*the file doesn't exist*)
		$Failed,
		(*ELSE*)
		(*the file exists and we can continue - note that we should expand the filename first, as it only works with absolute file names*)
		(
			res = iImportPacketCapture[ExpandFileName@path];
			If[MatchQ[res,_LibraryFunctionError],
				(*THEN*)
				(*failed to import it for some reason*)
				(
					Message[Import::fmterr,"PCAP"];
					$Failed
				),
				(*ELSE*)
				(*good to go*)
				(
					{"Data"->TINSLink`Private`fixDataset@DatasetFileFrom[res]}
				)
			]
		)
	]
];

End[]

EndPackage[]