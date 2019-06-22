System`Private`NewContextPath[{"System`","ExternalService`", "Charting`"}];

(Unprotect[#];Clear[#];)&/@{
	System`AnatomyStyling,
	System`AnatomyPlot3D,
	System`AnatomySkinStyle
};

Unprotect["System`AnatomyPlot3D`Private`*"]
ClearAll["System`AnatomyPlot3D`Private`*"]

Begin["AnatomyPlot3D`Private`"]

quietedEntityValue[args__]:=Quiet[EntityValue[args], {EntityValue::conopen,EntityValue::nodat, EntityValue::outdcache}]

anatEntityQ[e_Entity] := 
 MatchQ[e[[1]], "AnatomicalStructure" | "AnimalAnatomicalStructure"]
anatEntityQ[_] := False

anatEntityClassQ[ec_EntityClass] := 
 MatchQ[ec[[1]], "AnatomicalStructure" | "AnimalAnatomicalStructure"]
anatEntityClassQ[_] := False

earlyAnatomyStylingRules = {
    AnatomyStyling[AnatomyStyling[x_]]:>AnatomyStyling[x],
    AnatomyStyling["XRay"]:>First["Directives" /. OptionValue[
    	Charting`ResolvePlotTheme["XRay", "AnatomyPlot3D"], 
    	Method]
    ]
};

customStyleRules = {
	"Natural" -> Inherited,	
	"Highlighted" -> Directive[Red],
	{"TransparentBacklit", o_, c_} :> Directive[Specularity[White, 50], Opacity[o], c, Lighting -> Join[{{"Ambient", Black}}, 
		Table[{"Directional", White, ImageScaled[{Sin[x], Cos[x], -.5}]}, {x, 0, 2 Pi - 2 Pi/8, 2 Pi/8}]]],
	"TransparentBacklit" -> Directive[Specularity[White, 50], Hue[.58, 0, 1, .1], Lighting -> Join[{{"Ambient", Black}}, 
		Table[{"Directional", Hue[.58, .5, 1], ImageScaled[{Sin[x], Cos[x], -.5}]}, {x, 0, 2 Pi - 2 Pi/8, 2 Pi/8}]]],
	"XRay"-> Directive[Specularity[White, 50], Hue[.58, 0, 1, .1], Lighting -> Join[{{"Ambient", GrayLevel[.5]}}, Table[{"Directional", White, ImageScaled[{Sin[x], Cos[x], -.5}]}, {x, 0, 2 Pi - 2 Pi/8, 2 Pi/8}]]]
};

anatomyFormRules = {
   System`AnatomyStyling["Natural"] :> {All -> Inherited},
   System`AnatomyStyling["Highlighted"] :> {All -> Replace["Highlighted", customStyleRules]},
   System`AnatomyStyling[{"TransparentBacklit", o_, c_}] :> {All -> Replace[{"TransparentBacklit", o, c}, customStyleRules]},
   System`AnatomyStyling["TransparentBacklit"] :> {All -> Replace["TransparentBacklit", customStyleRules]},
   System`AnatomyStyling["XRay"] :> {All -> Replace["XRay", customStyleRules]},
   System`AnatomyStyling[] :> {All -> FaceForm[]},
   System`AnatomyStyling[
     rules : {_Rule ..}] :> DeleteDuplicatesBy[
     MapAt[
     	If[Head[#] === Directive, #, Directive[#]] &, 
     	rules, 
     	{{All, 2}}
     ],
     First],
   System`AnatomyStyling[assoc_Association] :> 
    With[{rules = Normal[assoc, Association]/.Rule[Verbatim[_],val_]:>Rule[All, val]},
     DeleteDuplicatesBy[
     MapAt[
     	If[Head[#] === Directive, #, Directive[#]] &, 
     	rules, 
     	{{All, 2}}
     ],
     First]],
   System`AnatomyStyling[
     arg_List] :> (Reverse[
      DeleteDuplicates[
       Reverse[((If[MatchQ[#, _Rule], #, 
              All -> Replace[#, customStyleRules]] & /@ 
            Flatten[arg]) /. 
          Rule[entity_, dir_] :> 
           Rule[entity, 
            If[Head[dir] === Directive, dir, Directive[dir]]])], #1[[
          1]] === #2[[1]] &]]),
   System`AnatomyStyling[
     arg_] :> ({All -> Replace[arg, customStyleRules]})};

applyAnatomyStyling[e_Entity, af_AnatomyStyling] := 
 Module[{initModelandTissueTypes, initOpts, initGeometryAndDirectives, 
 	styleRules, subPartStructureTypeRules, 
 	initAnatomyStylingRules, newSubpart},
 	initAnatomyStylingRules = af /. anatomyFormRules;
  (* The following retrieves the 3D model and the tissue types of the 
  subarts in a single EntityValue call *)
  initModelandTissueTypes = 
   quietedEntityValue[e, {"Graphics3D", "ImageAtomsToTissueType"}];
  
  (* If the 3D model is Missing, exit immediately, 
  there is nothing that can be done *)
  If[! MatchQ[initModelandTissueTypes[[1]], _Graphics3D], 
   Return[initModelandTissueTypes[[1]]]];
  
  initOpts = Options[initModelandTissueTypes[[1]]];
  initGeometryAndDirectives = initModelandTissueTypes[[1, 1]];
  
  (* Break down each entity into its atomic parts, assign the directives to those subparts, *)
  (* deleting Missing atomic subparts *)
  styleRules = 
   With[{(* remove Rule with LHS of All since we can't get ImageAtoms of those *)
   	prunedInitAnatomyStylingRules = DeleteCases[initAnatomyStylingRules, Rule[All,_]]},
   	With[{ia = quietedEntityValue[prunedInitAnatomyStylingRules[[All, 1]], "ImageAtoms"]},
   		If[MatchQ[Length[prunedInitAnatomyStylingRules[[All, 1]]], Length[ia]],
   			With[{imageAtomRules = (Rule @@@ Transpose[{prunedInitAnatomyStylingRules[[All, 1]], ia}]) /. 
   				Rule[ent_, _Missing] :> Rule[ent, ent]},
   				Flatten[(initAnatomyStylingRules /. ent_Entity :> (ent /. imageAtomRules)) /. 
   					Rule[ents : {_Entity ..}, dir_] :> (Rule[#, dir] & /@ ents)]], 
   			{}]]];
  (* Cleanup the above to remove All->Inherited inside of Directive *)
  styleRules = styleRules/.Rule[a_,Directive[All->Inherited]]:>Rule[a, Directive[Inherited]];
  (* Cleanup the above some more to remove Rule[All, Directive[Inherited,All->Inherited]] *)
  styleRules = DeleteCases[styleRules,Rule[All, Directive[Inherited,All->Inherited]]];
       
  If[styleRules === Rule[All, Inherited], 
   Return[
    Graphics3D[DeleteCases[initGeometryAndDirectives, Rule[All, Inherited]], 
     Sequence @@ initOpts]]];
  subPartStructureTypeRules = 
   With[{tmp = initModelandTissueTypes[[2]]}, 
    If[MatchQ[
      tmp, _Missing], {e -> (quietedEntityValue[e, 
          "StructureType"] /. {x_} :> x)}, tmp]];
  
  (* Reconstruct a new Graphics3D by walking through each subpart and seeing 
  if it needs to be modified *)
  Graphics3D[
   (* map over all subparts *)
   (Function[modelSubpart,
       With[{subpartEntity = modelSubpart[[-1, 2]]["Entity"], 
         subpartStructureType = 
          modelSubpart[[-1, 2]]["Entity"] /. 
           subPartStructureTypeRules},
        newSubpart = modelSubpart;
        (* for each subpart, 
        map over all the AnatomyStyling rules to see if any need applied, 
        checking both the TissueType and the entity itself *)
        Function[anatomyFormRule,
          If[(
          	((subpartStructureType === anatomyFormRule[[1]]) || 
          	 (subpartEntity === anatomyFormRule[[1]]))&&
             (!MatchQ[anatomyFormRule[[2]], Directive[Inherited]])
             ) ||
            ((anatomyFormRule[[1]]===All)&&
             (!((subpartStructureType === anatomyFormRule[[1]]) || 
             (subpartEntity === anatomyFormRule[[1]])))&&
             (!MatchQ[anatomyFormRule[[2]], Directive[Inherited]]) &&
             (FreeQ[styleRules, subpartEntity])
             ),
           newSubpart = {newSubpart[[1]] /. 
              Directive[d___] :> 
               (Directive[d, Sequence @@ anatomyFormRule[[2]]]/.customStyleRules), 
             newSubpart[[2]]}]] /@ styleRules; 
        newSubpart]] /@ 
      initGeometryAndDirectives) /. (Rule[All, Inherited]) :> (Sequence @@ {}), 
   Sequence @@ initOpts]
  ]

evaluateAnatomyStyling[anatomyInput_List] := Module[{styleF, result},
  styleF[0] = System`AnatomyStyling["Natural"];
  result = recurseAF[anatomyInput, {styleF, 0}];
  Remove[styleF];
  result
  ]
Clear[recurseAF];
recurseAF[list_List, {styleF_, level_}] := Module[{},
   (* Start new list with the anatomy form of the previous level as default *)
   styleF[level + 1] = styleF[level];
   recurseAF[#, {styleF, level + 1}] & /@ list
   ];
recurseAF[GraphicsGroup[gg_, opts:OptionsPattern[]], {styleF_, level_}] := Module[{},
   (* Start new list with the anatomy form of the previous level as default *)
   styleF[level + 1] = styleF[level];
   GraphicsGroup[recurseAF[#, {styleF, level + 1}] & /@ gg, opts]
   ];
recurseAF[Tooltip[arg1_, e_?anatEntityQ, opts:OptionsPattern[]], {styleF_, level_}] := Module[{},
   (* Start new list with the anatomy form of the previous level as default *)
   styleF[level + 1] = styleF[level];
   Tooltip[Sequence@@(recurseAF[#, {styleF, level + 1}] & /@ {arg1}), e["Name"], opts]
   ];
recurseAF[Tooltip[arg1_, opts:OptionsPattern[]], {styleF_, level_}] := Module[{},
   (* Start new list with the anatomy form of the previous level as default *)
   styleF[level + 1] = styleF[level];
   With[{seq=Sequence@@(recurseAF[#, {styleF, level + 1}] & /@ {arg1})},
   	Tooltip[seq, arg1, opts]]
   ];
recurseAF[Tooltip[arg1_, arg2_, opts:OptionsPattern[]], {styleF_, level_}] := Module[{},
   (* Start new list with the anatomy form of the previous level as default *)
   styleF[level + 1] = styleF[level];
   Tooltip[Sequence@@(recurseAF[#, {styleF, level + 1}] & /@ {arg1}), arg2, opts]
   ];
recurseAF[Rotate[arg1_, {u_?VectorQ,v_?VectorQ}], {styleF_, level_}] := Module[{},
   (* Start new list with the anatomy form of the previous level as default *)
   styleF[level + 1] = styleF[level];
   Rotate[Sequence@@(recurseAF[#, {styleF, level + 1}] & /@ {arg1}), {u,v}]
   ];
recurseAF[Rotate[arg1_, theta_, w_?VectorQ], {styleF_, level_}] := Module[{},
   (* Start new list with the anatomy form of the previous level as default *)
   styleF[level + 1] = styleF[level];
   Rotate[Sequence@@(recurseAF[#, {styleF, level + 1}] & /@ {arg1}), theta, w]
   ];
recurseAF[Rotate[arg1_, theta_, w_?VectorQ, p_?VectorQ], {styleF_, level_}] := Module[{},
   (* Start new list with the anatomy form of the previous level as default *)
   styleF[level + 1] = styleF[level];
   Rotate[Sequence@@(recurseAF[#, {styleF, level + 1}] & /@ {arg1}), theta, w, p]
   ];
recurseAF[Rotate[arg1_, theta_, w_?VectorQ, p_?anatEntityQ], {styleF_, level_}] := Module[{},
   (* Start new list with the anatomy form of the previous level as default *)
   styleF[level + 1] = styleF[level];
   Rotate[Sequence@@(recurseAF[#, {styleF, level + 1}] & /@ {arg1}), theta, w, p["RegionCentroid"]]
   ];
recurseAF[Rotate[arg1_, theta_, {u_?VectorQ, v_?VectorQ}], {styleF_, level_}] := Module[{},
   (* Start new list with the anatomy form of the previous level as default *)
   styleF[level + 1] = styleF[level];
   Rotate[Sequence@@(recurseAF[#, {styleF, level + 1}] & /@ {arg1}), theta, {u, v}]
   ];
recurseAF[Scale[arg1_, arg2__], {styleF_, level_}] := Module[{},
   (* Start new list with the anatomy form of the previous level as default *)
   styleF[level + 1] = styleF[level];
   Scale[Sequence@@(recurseAF[#, {styleF, level + 1}] & /@ {arg1}), arg2]
   ];
recurseAF[GeometricTransformation[arg1_, arg2__], {styleF_, level_}] := Module[{},
   (* Start new list with the anatomy form of the previous level as default *)
   styleF[level + 1] = styleF[level];
   GeometricTransformation[Sequence@@(recurseAF[#, {styleF, level + 1}] & /@ {arg1}), arg2]
   ];
recurseAF[Translate[arg1_, arg2_], {styleF_, level_}] := Module[{},
   (* Start new list with the anatomy form of the previous level as default *)
   styleF[level + 1] = styleF[level];
   Translate[Sequence@@(recurseAF[#, {styleF, level + 1}] & /@ {arg1}), arg2]
   ];
recurseAF[Inset[arg1_, opts:OptionsPattern[]], {styleF_, level_}] := Module[{},
   (* Start new list with the anatomy form of the previous level as default *)
   styleF[level + 1] = styleF[level];
   Inset[Sequence@@(recurseAF[#, {styleF, level + 1}] & /@ {arg1}), opts]
   ];
recurseAF[Inset[arg1_, arg2_?anatEntityQ, arg3___], {styleF_, level_}] := Module[{},
   (* Start new list with the anatomy form of the previous level as default *)
   styleF[level + 1] = styleF[level];
   Inset[Sequence@@(recurseAF[#, {styleF, level + 1}] & /@ {arg1}), arg2["RegionCentroid"], arg3]
   ];
recurseAF[Annotation[arg1_, arg2__], {styleF_, level_}] := Module[{},
   (* Start new list with the anatomy form of the previous level as default *)
   styleF[level + 1] = styleF[level];
   Annotation[Sequence@@(recurseAF[#, {styleF, level + 1}] & /@ {arg1}), arg2]
   ];
recurseAF[Button[arg1_, arg2__], {styleF_, level_}] := Module[{},
   (* Start new list with the anatomy form of the previous level as default *)
   styleF[level + 1] = styleF[level];
   Button[Sequence@@(recurseAF[#, {styleF, level + 1}] & /@ {arg1}), arg2]
   ];
recurseAF[EventHandler[arg1_, arg2__], {styleF_, level_}] := Module[{},
   (* Start new list with the anatomy form of the previous level as default *)
   styleF[level + 1] = styleF[level];
   EventHandler[Sequence@@(recurseAF[#, {styleF, level + 1}] & /@ {arg1}), arg2]
   ];
recurseAF[Hyperlink[arg1_, arg2__], {styleF_, level_}] := Module[{},
   (* Start new list with the anatomy form of the previous level as default *)
   styleF[level + 1] = styleF[level];
   Hyperlink[Sequence@@(recurseAF[#, {styleF, level + 1}] & /@ {arg1}), arg2]
   ];
recurseAF[Mouseover[arg1_, arg2__], {styleF_, level_}] := Module[{},
   (* Start new list with the anatomy form of the previous level as default *)
   styleF[level + 1] = styleF[level];
   Mouseover[Sequence@@(recurseAF[#, {styleF, level + 1}] & /@ {arg1}), Sequence@@(recurseAF[#, {styleF, level + 1}] & /@ {arg2})]
   ];
recurseAF[PopupWindow[arg1_, arg2__], {styleF_, level_}] := Module[{},
   (* Start new list with the anatomy form of the previous level as default *)
   styleF[level + 1] = styleF[level];
   PopupWindow[Sequence@@(recurseAF[#, {styleF, level + 1}] & /@ {arg1}), arg2]
   ];
recurseAF[StatusArea[arg1_, arg2__], {styleF_, level_}] := Module[{},
   (* Start new list with the anatomy form of the previous level as default *)
   styleF[level + 1] = styleF[level];
   StatusArea[Sequence@@(recurseAF[#, {styleF, level + 1}] & /@ {arg1}), arg2]
   ];
recurseAF[af_AnatomyStyling, {styleF_, level_}] := (
   styleF[level] = mergeStyles[styleF[level], af];
   Nothing
   );
recurseAF[dir_?directiveQ, {styleF_, level_}] := (
   styleF[level] = 
    mergeStyles[styleF[level], System`AnatomyStyling[All -> dir]];
   dir
   );

recurseAF[prim_, {styleF_, level_}] := prim;

directiveQ[dir_] := System`Dump`ValidDirective @@ {dir}

recurseAF[
   entity_?anatEntityQ, {styleF_, level_}] := 
  applyAnatomyStyling[entity, styleF[level]];

mergeStyles[System`AnatomyStyling[dir1___], 
  System`AnatomyStyling[dir2___]] := Module[{cleanupDir1, cleanupDir2, convertEverythingToRules, expandEntityClasses, gatherRulesByLHS, mergeResultsAndVerifyRules, getRidOfSimpleDirectiveHeads, cleanupInheriteds},
  (* If it's not already a Rule, then it applies to All structures so turn it into such a rule *)
  (* temporarily change custom directive heads away from List heads to C to avoid damage from later Flatten *)
  (* Charting`ConstructDirective validates and doesn't like custom styles, but passes through rules, so convert custom styles to rules *)
  cleanupDir1 = Flatten@If[dir1===Null, 
  	{All -> FaceForm[]},
  	{(Replace[Replace[Normal[dir1], {
  		{"TransparentBacklit", args__} :> Rule["AnatomySpecialStyle", C["TransparentBacklit",args]]
  		}], {
  		"Natural" :> Rule["AnatomySpecialStyle", "Natural"],
  		"Highlighted" :> Rule["AnatomySpecialStyle", "Highlighted"],
  		"XRay" :> Rule["AnatomySpecialStyle", "XRay"],
  		"TransparentBacklit" :> Rule["AnatomySpecialStyle", "TransparentBacklit"]
  		}, {2}]/.Rule["AnatomySpecialStyle", C[Rule["AnatomySpecialStyle", b_], c__]] :> Rule["AnatomySpecialStyle",C[b,c]])/.Rule[Verbatim[_],val_]:>Rule[All, val]}
  	];
  cleanupDir2 = Flatten@If[dir2===Null, 
  	{All -> FaceForm[]},
  	{(Replace[Replace[Normal[dir2], {
  		{"TransparentBacklit", args__} :> Rule["AnatomySpecialStyle", C["TransparentBacklit",args]]
  		}], {
  		"Natural" :> Rule["AnatomySpecialStyle", "Natural"],
  		"Highlighted" :> Rule["AnatomySpecialStyle", "Highlighted"],
  		"XRay" :> Rule["AnatomySpecialStyle", "XRay"],
  		"TransparentBacklit" :> Rule["AnatomySpecialStyle", "TransparentBacklit"]
  		}, {2}]/.Rule["AnatomySpecialStyle", C[Rule["AnatomySpecialStyle", b_], c__]] :> Rule["AnatomySpecialStyle",C[b,c]])/.Rule[Verbatim[_],val_]:>Rule[All, val]}
  	];
  (* The following code can generate things like All->AnatomySpecialStyle->AnatomySpecialStyle->Inherited so care must be taken to collapse this down *)
  convertEverythingToRules = (If[MatchQ[#,_Rule]&&!MatchQ[#, Rule["AnatomySpecialStyle", _]], 
  	Rule[#[[1]], Replace[Replace[#[[2]], {
  		"Natural" -> Rule["AnatomySpecialStyle", Inherited],
  		"Highlighted" -> Rule["AnatomySpecialStyle", "Highlighted"],
  		{"TransparentBacklit", args__} :> Rule["AnatomySpecialStyle", C["TransparentBacklit",args]],
  		"TransparentBacklit" -> Rule["AnatomySpecialStyle", "TransparentBacklit"]
  		}], Rule[a_, Rule[a_, b_]] :> Rule[a,b]]], 
  	Rule[All, Replace[Replace[#, {
  		"Natural" -> Rule["AnatomySpecialStyle", Inherited],
  		"Highlighted" -> Rule["AnatomySpecialStyle", "Highlighted"],
  		{"TransparentBacklit", args__} :> Rule["AnatomySpecialStyle", C["TransparentBacklit",args]],
  		"TransparentBacklit" -> Rule["AnatomySpecialStyle", "TransparentBacklit"]
  	}], Rule[a_, Rule[a_, b_]] :> Rule[a,b]]]]&/@Flatten[{cleanupDir1, cleanupDir2}]);
  expandEntityClasses = convertEverythingToRules/. Rule[ec_?anatEntityClassQ, style_]:> Sequence@@(Rule[#,style]&/@EntityList[ec]);
  gatherRulesByLHS = GatherBy[expandEntityClasses,#[[1]]&];
  mergeResultsAndVerifyRules = (Rule[#[[1,1]], Charting`ConstructDirective[#[[All,2]]]]&/@gatherRulesByLHS);
  getRidOfSimpleDirectiveHeads = mergeResultsAndVerifyRules /. Directive[x_]:>x;
  cleanupInheriteds = (getRidOfSimpleDirectiveHeads /. Directive[x___, Rule["AnatomySpecialStyle", Inherited], y___] :> Directive[y] /; FreeQ[{y}, Inherited])/.Directive[]:>Inherited;
  (* restore custom styles back to correct forms *)
  System`AnatomyStyling[
  	Replace[cleanupInheriteds,
  		{
  			HoldPattern[Rule["AnatomySpecialStyle", Inherited]] :> "Natural",
  			HoldPattern[Rule["AnatomySpecialStyle", "Highlighted"]] :> "Highlighted",
  			HoldPattern[Rule["AnatomySpecialStyle", "XRay"]] :> "XRay",
  			HoldPattern[Rule["AnatomySpecialStyle", C["TransparentBacklit",args__]]] :> {"TransparentBacklit", args},
  			HoldPattern[Rule["AnatomySpecialStyle", "TransparentBacklit"]] :> "TransparentBacklit"
  		}, {2}]/.C->List]]

Clear[System`AnatomyPlot3D]

anatomyOptionDefaults = {
	System`AnatomySkinStyle -> None,
	PlotTheme :> $PlotTheme
}

Options[System`AnatomyPlot3D] = SortBy[
		Join[
			Options[Graphics3D],
			anatomyOptionDefaults
		], ToString
	];

SetOptions[System`AnatomyPlot3D, {Boxed -> False, Lighting -> "Neutral", ViewPoint -> {0,-1.9,0}}];

duplicateFilter[opts___] :=
 DeleteCases[FilterRules[Flatten[{opts, FilterRules[Options[System`AnatomyPlot3D], Except[Flatten[{opts}]]]}],
  Options[Graphics3D]], Alternatives@@DeleteCases[Options[Graphics3D], Alternatives@@List@@opts]]
  
(*nickl note: statically assigning this rather than re-generating every time iAnatomyPlot3D is called*)
$grules = {
	Text[str_, rest__] :> 
     Text[str, Sequence@@({rest} /. {e_?anatEntityQ :> 
          e["RegionCentroid"]})],
    Polygon[p_, opts___] :> 
     Polygon[(p /. {e_?anatEntityQ :> 
          e["RegionCentroid"]}), opts],
    Line[l_, opts___] :> 
     Line[(l /. {e_?anatEntityQ :> 
          e["RegionCentroid"]}), opts],
    Tube[p1_, 
      p2___, opts:OptionsPattern[]] :> (Tube[
      p1 /. {e_?anatEntityQ :> e["RegionCentroid"]}, 
      Sequence@@({p2} /. {e_?anatEntityQ :> (e["EnclosingSphere"][[2]])}), opts]),
    Arrow[Tube[p1_, 
      p2___, opts:OptionsPattern[]], p3___] :> Arrow[Tube[
      p1 /. {e_?anatEntityQ :> e["RegionCentroid"]}, 
      Sequence@@({p2} /. {e_?anatEntityQ :> (e["EnclosingSphere"][[2]])}), opts], 
      Sequence@@({p3} /. {e_?anatEntityQ :> (e["EnclosingSphere"][[2]])})],
    Arrow[(h:Line|BSplineCurve|BezierCurve)[p1_, opts:OptionsPattern[]], p2___] :> Arrow[h[
      p1 /. {e_?anatEntityQ :> e["RegionCentroid"]}, opts], 
      Sequence@@({p2} /. {e_?anatEntityQ :> (e["EnclosingSphere"][[2]])})],
    Arrow[l_,setback___] :> 
     Arrow[(l /. {e_?anatEntityQ :> 
          e["RegionCentroid"]}), Sequence@@({setback} /. {e_?anatEntityQ :> (e["EnclosingSphere"][[2]])})],
    Point[p_, opts___] :> 
     Point[p /. {e_?anatEntityQ :> 
         e["RegionCentroid"]}, opts],
    Sphere[centers_, r___] :> 
     Sphere[(centers /. {e_?anatEntityQ :> 
          e["RegionCentroid"]}), Sequence@@({r} /. {e_?anatEntityQ :> (e["EnclosingSphere"][[2]])})],
    SphericalShell[centers_, r___] :> 
     SphericalShell[(centers /. {e_?anatEntityQ :> 
          e["RegionCentroid"]}), Sequence@@({r} /. {e_?anatEntityQ :> (e["EnclosingSphere"][[2]])})],
    Ball[centers_, r___] :> 
     Ball[(centers /. {e_?anatEntityQ :> 
          e["RegionCentroid"]}), Sequence@@({r} /. {e_?anatEntityQ :> (e["EnclosingSphere"][[2]])})],
    Cylinder[p1_, 
      p2___] :> (Cylinder[
      p1 /. {e_?anatEntityQ :> e["RegionCentroid"]}, 
      Sequence@@({p2} /. {e_?anatEntityQ :> (e["EnclosingSphere"][[2]])})]),
    Cuboid[p1_, 
      p2___] :> (Cuboid[p1, 
        p2] /. {e_?anatEntityQ :> 
         e["RegionCentroid"]}),
    Parallelepiped[p_, v_] :> 
     Parallelepiped[(p /. {e_?anatEntityQ :> 
          e["RegionCentroid"]}), v /. {e_?anatEntityQ :> 
          e["RegionCentroid"]}], 
    Hexahedron[p_, opts___] :> 
     Hexahedron[(p /. {e_?anatEntityQ :> 
          e["RegionCentroid"]}), opts],
    Tetrahedron[p_, opts___] :> 
     Tetrahedron[(p /. {e_?anatEntityQ :> 
          e["RegionCentroid"]}), opts],
    Triangle[g_, opts___] :> 
     Triangle[(g/.{e_?anatEntityQ :> e["RegionCentroid"]}), opts],
    Pyramid[p_, opts___] :> 
     Pyramid[(p /. {e_?anatEntityQ :> 
          e["RegionCentroid"]}), opts],
    Prism[p_, opts___] :> 
     Prism[(p /. {e_?anatEntityQ :> 
          e["RegionCentroid"]}), opts],
    Simplex[p_, opts___] :> 
     Simplex[(p /. {e_?anatEntityQ :> 
          e["RegionCentroid"]}), opts],
    Cone[p_, r___] :> 
     Cone[(p /. {e_?anatEntityQ :> 
          e["RegionCentroid"]}), Sequence@@({r} /. {e_?anatEntityQ :> (e["EnclosingSphere"][[2]])})],
    BezierCurve[p_, opts___] :> 
     BezierCurve[(p /. {e_?anatEntityQ :> 
          e["RegionCentroid"]}), opts],
    BSplineCurve[p_, opts___] :> 
     BSplineCurve[(p /. {e_?anatEntityQ :> 
          e["RegionCentroid"]}), opts],
    BSplineSurface[p_] :> 
     BSplineSurface[(p /. {e_?anatEntityQ :> 
          e["RegionCentroid"]})],
    Rule[Lighting, val_]:>Rule[Lighting, val/.(e_?anatEntityQ:>e["RegionCentroid"])],
    AffineHalfSpace[g__] :> 
     AffineHalfSpace[Sequence@@({g}/.{e_?anatEntityQ :> e["RegionCentroid"]})],
    AffineSpace[g__] :> 
     AffineSpace[Sequence@@({g}/.{e_?anatEntityQ :> e["RegionCentroid"]})],
    ConicHullRegion[g__] :> 
     ConicHullRegion[Sequence@@({g}/.{e_?anatEntityQ :> e["RegionCentroid"]})],
    CapsuleShape[g_, rest_] :> 
     CapsuleShape[(g/.{e_?anatEntityQ :> e["RegionCentroid"]}), rest],
    GraphicsComplex[g_, rest_, opts___] :> 
     GraphicsComplex[(g/.{e_?anatEntityQ :> e["RegionCentroid"]}), rest, opts],
    Ellipsoid[g_, rest_] :> 
     Ellipsoid[(g/.{e_?anatEntityQ :> e["RegionCentroid"]}), rest /.{e_?anatEntityQ :> (e["EnclosingSphere"][[2]])}],
    HalfLine[g__] :> 
     HalfLine[Sequence@@({g}/.{e_?anatEntityQ :> e["RegionCentroid"]})],
    HalfPlane[g__] :> 
     HalfPlane[Sequence@@({g}/.{e_?anatEntityQ :> e["RegionCentroid"]})],
    Hyperplane[g__] :> 
     Hyperplane[Sequence@@({g}/.{e_?anatEntityQ :> e["RegionCentroid"]})],
    HalfSpace[g__] :> 
     HalfSpace[Sequence@@({g}/.{e_?anatEntityQ :> e["RegionCentroid"]})],
    Text[Style[g_?anatEntityQ, rest1__], rest2__] :> Text[Style[g, rest1], rest2],
    (* Handle Style for convenience at SW insistence, jfultz opposition, per mtrott plan to leave undocumented *) 
    Style[g_?anatEntityQ, rest__] :> {If[Length[{rest}]>1, AnatomyStyling[{rest}], AnatomyStyling[rest]], g},
    Style[g:{_?anatEntityQ..}, rest__] :> {If[Length[{rest}]>1, AnatomyStyling[{rest}], AnatomyStyling[rest]], g}
};

Clear[System`AnatomyPlot3D];
Clear[iAnatomyPlot3D];
iAnatomyPlot3D[anatomyInput_, opts : OptionsPattern[System`AnatomyPlot3D]] :=
Block[{System`AnatomyForm = System`AnatomyStyling, System`SkinStyle = System`AnatomySkinStyle}, 
 Module[{expandedFirstArgument, inputEntities0, inputEntities, optEntities, fullInputEntities, finalOpts, skinStyleValue, 
 	iAnatomyInput, models, entityModelPairs, missingModels, theme, o, methodRules, themeDirectives, entitiesToAdd, 
 	postProcessingRules, premethod, tooltip},
 	
  iAnatomyInput = If[MatchQ[anatomyInput, _List], anatomyInput, {anatomyInput}];
 	
  o = Flatten[{opts}];
  Quiet[theme = OptionValue[System`AnatomyPlot3D, o, PlotTheme]];
  theme = ResolvePlotTheme[theme, System`AnatomyPlot3D];
  Quiet[ premethod = OptionValue[System`AnatomyPlot3D, theme, Method]];
  premethod = ConstructMethod[ premethod];
  theme = FilterRules[theme, Join[Options[System`AnatomyPlot3D], theme]];
  o = Join[o, theme];
  
  methodRules = Quiet[OptionValue[{Method -> {}}, o, Method], {OptionValue::nodef}];
  methodRules = parseMethod[ConstructMethod[methodRules], premethod];
  {themeDirectives, postProcessingRules, entitiesToAdd, tooltip} = Quiet[
  	OptionValue[{"Directives" -> Nothing, "PostProcessing" -> {}, "Entities" -> Nothing, "Tooltips" -> Automatic}, 
  		methodRules, {"Directives", "PostProcessing", "Entities", "Tooltips"}], 
  		{OptionValue::nodef}
  	];
  Which[
  	tooltip===True, postProcessingRules = Prepend[postProcessingRules, Annotation[x_, y_Association] /; ! FreeQ[Keys[y], "Source"] :> Tooltip[Annotation[x, y], y["Name"]]],
  	tooltip===False, postProcessingRules = Prepend[postProcessingRules, Annotation[x_, y_Association] /; ! FreeQ[Keys[y], "Source"] :> Annotation[x, y]]
  ];
  inputEntities0 = Cases[iAnatomyInput, _Entity, Infinity];
		
  iAnatomyInput = Join[{Sequence@@(themeDirectives /. {"IndexedColor", num_}:> AnatomyStyling[
  		Flatten[MapIndexed[Rule[#1, ColorData[num][First[#2]]] &, #["ImageAtoms"]] & /@ inputEntities0]]), 
  		Sequence@@entitiesToAdd}, iAnatomyInput]/.eceaf:Except[_AnatomyStyling]:>ReplaceAll[eceaf, ec_?anatEntityClassQ:>(Sequence@@EntityList[ec])];
  
  (* expand the 1st argument, including the AnatomyStyling directives and handle any nested AnatomyStylings *)
  expandedFirstArgument = evaluateAnatomyStyling[Quiet[{iAnatomyInput /. $grules}, {EntityValue::conopen,EntityValue::nodat, EntityValue::outdcache}]/.earlyAnatomyStylingRules];
  inputEntities = Cases[iAnatomyInput, _Entity, Infinity];
  optEntities = Cases[{o}, _Entity, Infinity];
  (* re-fetch models, should be cached, to see if any are missing *)
  fullInputEntities = DeleteDuplicates[Join[inputEntities, optEntities]];
  models = quietedEntityValue[fullInputEntities, "Graphics3D"];
  entityModelPairs = Transpose[{fullInputEntities, models}];
  missingModels = Cases[entityModelPairs, {e_Entity, _Missing}:>e];
  Message[MessageName[AnatomyPlot3D, "missmod"], HoldForm[#]]&/@missingModels;
  (* the following strips out the explicit options of Graphics3D. No need to bake in the results *)
  (* unless the options are different than the default Graphics3D options. Also, allows the PlotRange to support *)
  (* an Entity spec and sorts the list of options *)
  finalOpts = 
   DeleteCases[duplicateFilter[o]/.{
   	Rule[PlotRange, e_?anatEntityQ] :> Rule[PlotRange, e["RegionBounds"]/._Missing :> Automatic], 
   	Rule[Lighting, val_]:> With[{rc=val/.(e_?anatEntityQ:>e["RegionCentroid"])}, 
   		Rule[Lighting, If[FreeQ[rc, _Missing], rc, "Neutral"]]],
   	Rule[SphericalRegion, e_?anatEntityQ]:> With[{val = e["EnclosingSphere"]}, 
   		Rule[SphericalRegion, If[FreeQ[val, _Missing], val, False]]],
   	Rule[SphericalRegion, Sphere[e_?anatEntityQ, r_?NumberQ]]:> With[{val = e["RegionCentroid"]}, 
   		Rule[SphericalRegion, If[FreeQ[val, _Missing], Sphere[val, r], False]]],
   	Rule[SphericalRegion, Sphere[e_?anatEntityQ, r_Quantity]]:> With[{val = e["RegionCentroid"]}, 
   		Rule[SphericalRegion, If[FreeQ[val, _Missing], Sphere[val, QuantityMagnitude[r, "Millimeters"]], False]]],
   	Rule[ViewVector, val_]:>With[{rc = val/.(e_?anatEntityQ:>e["RegionCentroid"])},
   		Rule[ViewVector, If[FreeQ[rc, _Missing], rc, Automatic]]]
   	},Alternatives@@DeleteCases[Options[Graphics3D], Alternatives@@List@@o]];
  
  Block[{Inherited = (Sequence @@ {})},
  	(* First check if the SkinStyling option needs to be applied *)
  	skinStyleValue = System`AnatomySkinStyle/.o;
  	If[!MatchQ[skinStyleValue, None]&&!MatchQ[skinStyleValue, System`AnatomySkinStyle],
   With[{
   	g3D=Graphics3D[(expandedFirstArgument /. Graphics3D[x_, ___] :> x), finalOpts], 
   	asp = DeleteDuplicates[DeleteMissing[quietedEntityValue[inputEntities0, "AssociatedSkinPart"]]]},
   	Graphics3D[({g3D[[1]],
   		If[MatchQ[asp, {_Entity..}],
   		(#[[1]]&/@(quietedEntityValue[asp, "Graphics3D"]))/.Directive[x__]:>Directive[x,
   		If[MatchQ[skinStyleValue, Automatic], 
   		Opacity[.3], 
   		If[directiveQ[skinStyleValue], skinStyleValue, Sequence@@{}]]],{}]}/._Missing->Sequence@@{})/.postProcessingRules, finalOpts]],
   Graphics3D[((expandedFirstArgument /. Graphics3D[x_, ___] :> x)/._Missing->Sequence@@{})/.postProcessingRules, finalOpts]]]
  ]
 ]

iAnatomyPlot3D[opts : OptionsPattern[System`AnatomyPlot3D]] := Graphics3D[{},
	duplicateFilter[opts]]

System`AnatomyPlot3D[args___, 
   opts : OptionsPattern[System`AnatomyPlot3D]] /; (ArgumentCountQ[System`AnatomyPlot3D, 
    Length[DeleteCases[{args}, _Rule, Infinity]], 0, 1]) := 
 Block[{res},
   res = iAnatomyPlot3D[args, Sequence@@Flatten[{opts}]];
   res /; ! MatchQ[res, _Missing | $Failed]] /; 
  FreeQ[{args}, _Rule, {1}]

End[];
  
SetAttributes[
	{AnatomyPlot3D, AnatomyStyling, AnatomySkinStyle},
	{ReadProtected, Protected}
];

  
  
System`Private`RestoreContextPath[];