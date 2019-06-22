(* Mathematica Package *)

(* $Id$ *)

BeginPackage["DeviceAPI`Drivers`Demos`WriteDemo`Dump`"]
(* Exported symbols added here with SymbolName::usage *)  

WriteDemo::prop = "The value of \"ReadType\" is must be one of Byte, Character or String.";  

Begin["`Private`"] (* Begin Private Context *)  

$fileName;
$str; 
$vals={};
$rule = {}; 
$readType = Automatic; 
 
openFun[___]:= ($fileName = FileNameJoin[{$TemporaryDirectory, "WriteDemoTestFile"}]; $str = OpenWrite[$fileName];)

writeFun[_,arg3___]:= (

						If[ Cases[Streams[],InputStream[$fileName, _]] != {},
						 
						   Close[$fileName] ];
						
						   PutAppend[arg3, $fileName]
						
						  )

readFun[args___]:= Module[{val, list},
					  
					  If[ Cases[Streams[],OutputStream[$fileName, _]] != {},
					   Close[$fileName] ]; 	  
					   val = read[$fileName, $readType];
					   
					   If[val === EndOfFile, 
					   	
						  (list = readList[$fileName,$readType];If[list==={}, Return[list],Return[Last@list]];), 
						  
						  Return[val]
						  ]
					   
					   ]

closeFun[___]:= (If[FileByteCount[$fileName]=== 0,Close[$fileName],DeleteFile[$fileName]]; $vals={};$rule = {}; $readType = Automatic;)
 
writeBufferFun[{arg1_,arg2_},vals_?ruleQ]:= ($rule = Join[$rule,{vals}];$rule)   

writeBufferFun[{arg1_,arg2_},vals___]:= writeBufferFun[{arg1,arg2},{vals}]

writeBufferFun[{arg1_,arg2_},vals_List]:= ($vals = Join[$vals,vals];$vals)(*writeFun[{arg1,arg2},arg3]*)
 
writeBufferFun[___]:=$Failed 
 
readBufferFun[{ihandle_,dhandle_},_]:= Join[$vals,$rule](*If[$vals==={},$rule,$vals]*)
 
readBufferFun[{ihandle_,dhandle_},n_Integer,_]:= With [{ret= Join[$vals,$rule]},
													  If[n>Length[ret],(Message[DeviceReadBuffer::blen,n];Return[$Failed]),Take[ret,n]]
													  ]
 
readBufferFun[{ihandle_,dhandle_},All | Automatic,param_]:= If[$rule === {}, {} ,param /. $rule]

readBufferFun[___]:=$Failed 
 
setProp[dev_, "ReadType",val_]/; !MemberQ[{Byte, Character, String, Automatic}, val] := (Message[WriteDemo::prop, val];Return@$Failed)
 
setProp[args___]:= (DeviceFramework`DeviceSetProperty[args]; With[{lst = {args}}, $readType = Last[lst]])
 
(****************************************************************************************************************************************************************) 
 
Attributes[ruleQ] = {HoldAll};
ruleQ[x_] := MatchQ[Unevaluated@x, _Rule | _RuleDelayed]  

read[arg1_,Automatic]:= Read[arg1]

read[arg1_,arg2_]:= Read[arg1,arg2] 
 
readList[arg1_,Automatic]:= ReadList[arg1]
 
readList[arg1_,arg2_]:= ReadList[arg1,arg2]  
 

(****************************************************************************************************************************************************************) 

DeviceFramework`DeviceClassRegister["WriteDemo",
	"OpenFunction"-> openFun,
	"WriteFunction"-> writeFun,
	"ReadFunction" -> readFun,
	"CloseFunction" :> closeFun,
	"WriteBufferFunction" -> writeBufferFun,
    "ReadBufferFunction" -> readBufferFun,
	"FindFunction" -> ({{False,{}}}&),
	"Singleton" -> True,
	"Properties" -> {"ReadType" -> Automatic},
	"SetPropertyFunction" -> setProp,
	"DeregisterOnClose" -> True,								
	"DriverVersion" -> 0.001
];

End[] (* End Private Context *)


EndPackage[]