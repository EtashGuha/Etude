(*******************************************************************************

bpe encoder creation and evaluation

*******************************************************************************)

Package["NumericArrayUtilities`"]

PackageImport["GeneralUtilities`"]
PackageImport["PacletManager`"]

(******************************************************************************)
DeclareLibraryFunction[initializeBPEProcessor, "bpe_sentencepieceprocessor_initialize", 
	LinkObject (* processor id, pieces, vocab, precompiled_charsmap, options *)
	, 
	LinkObject		(* Void *)			
]

DeclareLibraryFunction[encodeBPE, "bpe_sentencepieceprocessor_encode", 
	LinkObject (* processor id, sentences *)
	, 
	LinkObject     (* flattened encoded id sequences, length list *)
]

DeclareLibraryFunction[decodeBPE, "bpe_sentencepieceprocessor_decode", 
	LinkObject (* processor id, flattened id sequences, length list *)
	, 
	LinkObject (* decoded strings *)	
]


(******************************************************************************)

loadProtobufLink[] := Block[
	{$ContextPath = {"System`"}}, 
	Needs["ProtobufLink`"]; 
	Clear[loadProtobufLink]
]

PackageExport["ImportSentencePieceModel"]

SetUsage[ImportSentencePieceModel, "
ImportSentencePieceModel[modelPath$, vocabPath$] imports a sentencepiece model file into an \
association and a vocabulary file into a string, returning the list {modelAssoc$, vocabString$}.
The vocabulary path can be None, in which case an empty string is returned. "]

ImportSentencePieceModel[modelPath_String, vocabPath:(_String | None)] := Scope[
	If[!FileExistsQ[modelPath], Panic["NoProtoFile", "protobuf file `` not found", modelPath]]; 
	If[MatchQ[vocabPath, _String] && !FileExistsQ[vocabPath], Panic["NoVocabFile", "vocabulary file `` not found", vocabPath]];
	loadProtobufLink[];
	proto = Quiet @ ProtobufLink`ProtobufImport[ 
		PacletResource["NumericArrayUtilities", "SentencePieceModelProto"], 
		modelPath, 
		"sentencepiece.ModelProto" 
	]; 
	vocab = Replace[vocabPath,
		{
			s_String :> Quiet @ Import[s, "Text"],
			None -> ""
		}
	];
	{proto, vocab}
]

PackageExport["BPEProcessor"]

SetUsage[BPEProcessor, "
BPEProcessor[id$] is a managed library expression for a SentencepieceProcessor object with algorithm  \
set to BPE."]

PackageExport["CreateBPEProcessor"]

SetUsage[CreateBPEProcessor, "
CreateBPEProcessor[pieces$, vocab$] creates, initialized and returns a BPEProcessor \
for BPE encoding/decoding with the desired specifications. The lists pieces$ and vocab$ contain \
respectively the BPE tokens and the vocabulary. Allowed elements for both lists are strings, \
StartOfString, EndOfString and the unkonwn token represented by _. 

The following options from sentencepiece are exposed. They modify the behaviour of the output \
processor:
| \"AddDummyPrefix\" | False | Add a whitespace at the beginning of the input string when encoding |
| \"RemoveExtraWhitespaces\" | False | Combine adjacent whitespace characters into one when encoding |
| \"EscapeWhitespaces\" | False | Replace whitespace characters with the sentencepiece special character |
| \"ReverseSentences\" | False | Return training history along with result |

\"EscapeWhitespaces\" will replace whitespace with character \"\\xe2\\x96\\x81\", which typesets as \
\"\:2581\" (WL code: 2581). \"EscapeWhitespaces\" -> True is not properly supported now, as it interacts badly \
with \"PreMergeWhitespaces\", and it's hidden from top level.

Our implementation also supports the additional options:
| \"AddDummyPostfix\" | False | Add a whitespace at the end of the input string when encoding |
| \"PreMergeWhitespace\" | None | Merge whitespace at the left or right of each word with the adjacent character before starting the bpe tokenization |

\"PreMergeWhitespace\" can be set to Left or Right to enable it. This functionality is used by other BPE implementations including the original one. For a motivation, \
see https://github.com/rsennrich/subword-nmt/issues/19.
"]

Options[CreateBPEProcessor] = {
	"AddDummyPrefix" -> False,
	"AddDummyPostfix" -> False,
	"InsertBOS" -> False,
	"InsertEOS" -> False,
	"RemoveExtraWhitespaces" -> False,
	"EscapeWhitespaces" -> False, (* Will currently break if this and PreMergeWhitespaces are True *)
	"ReverseSentences" -> False,
	"PreMergeWhitespace" -> None,
	"CaseFolding" -> False,
	"UTF8Normalization" -> None
}

preDefinedCharMaps := preDefinedCharMaps = BinaryDeserialize[
	Import @ PacletResource["NumericArrayUtilities", "PrecompiledCharsMap"]
]

CreateBPEProcessor::nounk = "Unknown character _ not found, will be added automatically.";

CreateBPEProcessor[pieces_, vocab_, OptionsPattern[]] := Scope[
	UnpackOptions[
		addDummyPrefix, 
		addDummyPostfix, 
		insertBOS,
		insertEOS,
		removeExtraWhitespaces, 
		escapeWhitespaces, 
		reverseSentences, 
		preMergeWhitespace,
		caseFolding,
		uTF8Normalization
	];

	instance = CreateManagedLibraryExpression["SentencePieceProcessor", BPEProcessor];

	pieces = Replace[pieces, 
		{StartOfString -> "<s>", EndOfString -> "</s>", Verbatim[_] -> "<unk>"},
		{1}
	];
	If[!MemberQ[pieces, "<unk>"], Message[CreateBPEProcessor::nounk]];
	If[vocab =!= None,
		vocab = Replace[vocab, 
			{StartOfString -> "<s>", EndOfString -> "</s>", Verbatim[_] -> "<unk>"},
			{1}
		],
		vocab = {}
	];

	charMap = Replace[{caseFolding, uTF8Normalization}, 
		{
			{False, "NFKC"} -> preDefinedCharMaps["nfkc"],
			{False, "ModifiedNFKC"} -> preDefinedCharMaps["nmt_nfkc"],
			{False, None} -> {},
			{True, "NFKC"} -> preDefinedCharMaps["nfkc_cf"],
			{True, "ModifiedNFKC"} -> preDefinedCharMaps["nmt_nfkc_cf"],
			{True, None} :> Panic["UnsupportedOptions", 
				"UTF8Normalization -> None together with CaseFolding -> True is not supported."
			],
			{_, arr_ByteArray} :> arr
		}
	];
	boolOpts = Replace[
		{addDummyPrefix, addDummyPostfix, removeExtraWhitespaces, escapeWhitespaces, insertBOS, insertEOS, reverseSentences}, 
		{True -> 1, False -> 0}, 
		{1}
	];
	preMergeWhitespace = Replace[preMergeWhitespace, {None -> 1, Left -> 2, Right -> 3}];
	options = Append[boolOpts, preMergeWhitespace];

	return = initializeBPEProcessor[
		ManagedLibraryExpressionID[instance], 
		pieces, 
		vocab, 
		Normal[charMap],
		options
	];
	If[MatchQ[return, _LibraryFunctionError], Return@$Failed];

	instance
]

PackageExport["EncodeBPE"]

SetUsage[EncodeBPE, "
EncodeBPE[processor$, input$] performs the BPE encoding of input$ using the BPE processor processor$. \
The input can be either a string or a list of strings."]

EncodeBPE[processor_, sentences:{Repeated[_String]}] := Scope[
	return = encodeBPE[
		sentences,
		ManagedLibraryExpressionID[processor]
	];
	If[MatchQ[return, _LibraryFunctionError], Return@$Failed];
	{flattened, lengthList} = return;
	TakeList @@ {flattened + 1, lengthList} (* convert to 1-index *)
]
EncodeBPE[processor_, sentence_String] := First@EncodeBPE[processor, {sentence}]

PackageExport["DecodeBPE"]

SetUsage[DecodeBPE, "
DecodeBPE[processor$, input$] performs the BPE decoding of input$ using the BPE processor processor$. \
The input can be either a list of integer ids or a batch of those."]

DecodeBPE[processor_, ids:{Repeated[_List]}] := Scope[
	joined = Join @@ ids;
	lengthList = Length /@ ids;
	return = decodeBPE[
		ManagedLibraryExpressionID[processor], 
		joined - 1, (* Convert to 0-index *) 
		lengthList
	];
	If[MatchQ[return, _LibraryFunctionError], Return@$Failed];
	return
]
DecodeBPE[processor_, ids:{Repeated[_Integer]}] := First@DecodeBPE[processor, {ids}]