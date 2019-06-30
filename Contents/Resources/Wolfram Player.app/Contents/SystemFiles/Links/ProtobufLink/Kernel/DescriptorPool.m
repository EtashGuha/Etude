(*******************************************************************************

Descriptor Pool

*******************************************************************************)

Package["ProtobufLink`"]

PackageImport["GeneralUtilities`"]

(*----------------------------------------------------------------------------*)
descriptorConstruct = LibraryFunctionLoad[$ProtoLinkLib, "WL_DescriptorPoolConstruct", 
		LinkObject,
		LinkObject		
]

(*----------------------------------------------------------------------------*)
PackageExport["ProtobufDescriptor"]
PackageScope["descriptorMLE"]

(* This is a utility function defined in GeneralUtilities, which makes a nicely
formatted display box *)
DefineCustomBoxes[ProtobufDescriptor, 
	e:ProtobufDescriptor[mle_] :> Block[{},
	BoxForm`ArrangeSummaryBox[
		ProtobufDescriptor, e, None, 
		{
			BoxForm`SummaryItem[{"ID: ", getMLEID[mle]}]
		},
		{},
		StandardForm
	]
]];

getMLE[ProtobufDescriptor[mle_]] := mle;
getMLEID[ProtobufDescriptor[mle_]] := ManagedLibraryExpressionID[mle];

(*----------------------------------------------------------------------------*)
PackageExport["ProtobufDescriptorCreate"]

SetUsage[
"ProtobufDescriptorCreate[file$] creates a ProtobufDescriptor[$$] object from a \
.proto file$. 
ProtobufDescriptorCreate[file$, paths$] creates a ProtobufDescriptor[$$] object \
using the list of file paths$ to search for any import dependencies.
"
]

ProtobufDescriptorCreate::protobufinvdir = 
	"`` passed as an import path is not a directory.";

ProtobufDescriptorCreate[src_, paths_List] := CatchFailure @ Module[
	{src2, mle, paths2}
	,
	src2 = fileConform[src];
	paths2 = fileConform /@ paths;
	If[!DirectoryQ[#], ThrowFailure[ProtobufDescriptorCreate::protobufinvdir, #]]& /@
		paths2;

	paths2 = Union[paths2, {DirectoryName[src2]}];
	src2 = FileNameTake[src2];

	mle = CreateManagedLibraryExpression["ProtoDescriptor", descriptorMLE];


	safeLibraryInvoke[descriptorConstruct,
		getMLEID[mle],
		src2, 
		paths2
	];

	System`Private`SetNoEntry @ ProtobufDescriptor[mle]
]

ProtobufDescriptorCreate[src_, path_String] := ProtobufDescriptorCreate[src, {path}]
ProtobufDescriptorCreate[src_, path_File] := ProtobufDescriptorCreate[src, {path}]
ProtobufDescriptorCreate[source_] := ProtobufDescriptorCreate[source, {}]