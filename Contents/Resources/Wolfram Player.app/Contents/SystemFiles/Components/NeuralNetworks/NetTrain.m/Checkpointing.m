Package["NeuralNetworks`"]


(* parsing of Checkpoint option  *)

parseCheckpointSpec[None] := Hold;

parseCheckpointSpec[spec_] := Replace[parseCS[spec], l_List :> ApplyThrough[l]];

NetTrain::invchkfile = "The value of the \"File\" parameter to TrainingProgressCheckpointing should specify a file in a directory that already exists."

parseCS[{"File", path_String, opts___Rule}] := Scope[
	path = ExpandFileName[path];
	If[!DirectoryQ[DirectoryName[path]], ThrowFailure["invchkfile"]];
	exporter = formatToExporter[FileExtension @ path];
	makePeriodicFunction[TrainingProgressCheckpointing, exporter /. # -> path, FilterOptions[opts]]
];

NetTrain::invchkext = "\"``\" is not a valid format for checkpointing. Only \"WLNet\" or \"params\" are supported."

formatToExporter[format_] := Switch[
	ToLowerCase[format],
	"wlnet", checkpointWrapper[WLNetExport[#, $currentNet]]&,
	"params", checkpointWrapper[TrainerSaveCheckpoint[$trainer, #]]&,
	_, ThrowFailure["invchkext", format]
];

SetHoldFirst[checkpointWrapper];
checkpointWrapper[body_] := 
	If[FreeQ[handleEvent["CheckpointStarted"], "SkipCheckpoint"],
		$CoreLoopLogger["CheckpointStarted"];
		body;
		handleEvent["CheckpointComplete"];
		$CoreLoopLogger["CheckpointComplete"];
	];

base10Digits[Infinity] := 7;
base10Digits[n_] := Floor[1 + Log[10, n]];

NetTrain::invchkdir = "The value of the \"Directory\" parameter to TrainingProgressCheckpointing should be a directory that already exists."
parseCS[{"Directory", path_String, opts___Rule}] := Scope[
	If[!DirectoryQ[path] && (!DirectoryQ[FileNameDrop[path]] || FailureQ[CreateDirectory[path]]),
		ThrowFailure["invchkdir"]
	];
	format = Lookup[{opts}, "Format", "wlnet"];
	exporter = formatToExporter[format];
	func = checkpointDir[
		path, exporter,
		dateStringForPath[] <> "_" <> IntegerString[$TrainingCounter++] <> "_",
		base10Digits[maxTrainingRounds], 
		base10Digits[$maxBatches],
		"." <> ToLowerCase[format]
	];
	makePeriodicFunction[TrainingProgressCheckpointing, func, FilterOptions[opts]]
];

dateStringForPath[] := 
	If[$OperatingSystem === "Windows", StringReplace[":" -> "-"], Identity] @ DateString["ISODateTime"];

parseCS[list_List ? ListOfListsQ] :=
	Map[parseCS, list];

NetTrain::invchkspec = "The value of the TrainingProgressCheckpointing option should be a spec of the form {\"File\"|\"Directory\", \"path\"}."
parseCS[spec_] := ThrowFailure["invchkspec", spec];

checkpointDir[path_, exportFunc_, startString_, roundBase_, batchBase_, ext_][] := Scope[
	filename = StringJoin[
		startString,
		IntegerString[$round, 10, roundBase], "_",
		IntegerString[$absoluteBatch, 10, batchBase], "_", lossString[$roundLoss],
		If[$doValidation, {"_", lossString[$validationLoss]}, {}], ext
	];
	filepath = FileNameJoin[{path, filename}];
	BagPush[$checkpointFiles, filepath];
	exportFunc[filepath];
];

lossString[_] := "none";
lossString[r_Real] := SciString[r];
