(* Mathematica Package *)

BeginPackage["CCodeGenerator`SystemsModel`", { "CCodeGenerator`"}]

SystemsModelCCodeGenerate::usage = "SystemsModelCCodeGenerate[ model, name, file] exports the ControlModel model as C code."
 

Begin["`Private`"] (* Begin Private Context *) 


SystemsModelCCodeGenerate[ (ss_StateSpaceModel)?DiscreteTimeModelQ, name_String, file_:Automatic, opts:OptionsPattern[]] :=
	SystemsModelCCodeGenerate[ ss, {name<>"_Output", name<>"_StateUpdate", name<>"_SamplingPeriod"}, file, opts]
	
SystemsModelCCodeGenerate[ (ss_StateSpaceModel)?DiscreteTimeModelQ, {name1_String, name2_String, name3_String}, file_:Automatic, opts:OptionsPattern[]] :=
	Module[ {c1, c2, c3},
		{c1,c2, c3} = makeControlObjectCode[ss];
		CCodeGenerate[ {c1,c2, c3}, {name1, name2, name3}, file, opts]
	]
	
	
	

makeControlObjectCode[(ss_StateSpaceModel)?DiscreteTimeModelQ] :=
 Module[{AMat, BMat, CMat, DMat},
  {AMat, BMat, CMat, DMat} = Normal[ss];
  With[{aM1 = N[AMat], bM1 = N[BMat], cM1 = N[CMat], dM1 = N[DMat], sample = N[Control`GetSamplingPeriod[ss]]},
   {
    Compile[{{xn, _Real, 1}, {un, _Real, 1}}, aM1.xn + bM1.un],
    Compile[{{xn, _Real, 1}, {un, _Real, 1}}, cM1.xn + dM1.un],
    Compile[ {}, sample]}
   ]] 



End[] (* End Private Context *)

EndPackage[]