(* $Id$ *)

BeginPackage["DeviceAPI`Drivers`Demos`StreamsDemo`Dump`"];

Begin["`Private`"];

(*----------- stream methods from tutorial/StreamMethods ----------*)

DefineInputStreamMethod["ByteList", {
  "ConstructorFunction" ->
   Function[{name, caller, opts}, 
    If[MatchQ[opts, {___, "Bytes" -> {_Integer ...}, ___}],
     {True, {0, "Bytes" /. opts}},
     {False, $Failed}
     ]],
  
  "ReadFunction" ->
   Function[{state, n},
    Module[{pos = state[[1]], bytes = state[[2]], bytesRead},
     If[pos >= Length[bytes],
      {{}, state},
      bytesRead = 
       Part[bytes, pos + 1 ;; Min[Length[bytes], pos + n]];
      {bytesRead, {pos + Length[bytesRead], bytes}}
      ]
     ]],
  
  "EndOfFileQFunction" -> ({#[[1]] >= Length[#[[2]]], #} &)
  }];

DefineOutputStreamMethod["Passthrough",
   {
      "ConstructorFunction" -> 
   Function[{streamname, isAppend, caller, opts},
    With[{state = Unique["PassthroughOutputStream"]},
     state["stream"] = OpenWrite[streamname, BinaryFormat -> True];
     state["pos"] = 0;
     {True, state}
     ]],
  
  "CloseFunction" -> 
   Function[state, Close[state["stream"]]; ClearAll[state]],
  
  "StreamPositionFunction" -> Function[state, {state["pos"], state}],
  
  "WriteFunction" ->
   Function[{state, bytes},
    Module[{result = BinaryWrite[state["stream"], bytes], nBytes},
     nBytes = If[result === state["stream"], Length[bytes], 0];
     Print["Wrote ", nBytes, " bytes"];
     state["pos"] += nBytes;
     {nBytes, state}
     ]
    ]
      }
  ];
  
firstWord[s_] := Module[{str = StringToStream[s],res},
	res = Read[str, Word];
	Close[str];
	res
]
		   

(* usage: DeviceOpen["StreamsDemo", {in,out}] or {{in1,in2...},{out1,out2...}} *)
	
openRead[_] := openRead[Null,{}]

openRead[_,i_,___] := Map[
	OpenRead[firstWord[#], Method -> {"ByteList", "Bytes" -> ToCharacterCode[#]}]&,
	ToString/@Flatten[{i}]
]


openWrite[_,_:{}] := openWrite[Null,Null,{}]

openWrite[_,_,o_] := Map[
	OpenWrite[
		FileNameJoin[{$TemporaryDirectory,#}],
		Method -> "Passthrough", BinaryFormat -> True
	]&,
	ToString/@Flatten[{o}]
]

setLabels[dev_] := DeviceFramework`DeviceStatusLabels[dev] =
{
	"Connected ("<>Sequence@@Riffle[
		FileNameTake/@First/@DeviceStreams[dev],
		", "
	]<>")",
	"Not connected"
}

(*-----------------------------------------------------------------*)  

DeviceFramework`DeviceClassRegister["StreamsDemo",
	"OpenReadFunction" -> openRead,
	"OpenWriteFunction" -> openWrite,
	"PreconfigureFunction" -> setLabels,
	"Singleton" -> True,
	"DeregisterOnClose" -> True,
	"DriverVersion" -> 0.001
];

End[];
EndPackage[];
