(** send graphics to Java via JLink **)

Begin["System`"]

$DisplayFunction
$SoundDisplayFunction

End[]

Begin["System`Private`"]

<<JLink`

(* graphics *)
JavaDisplay[gr_]:=
JavaBlock[Module[{frame, mathCanvas, pane, img, w, h},
  InstallJava[];
  frame = JavaNew["com.wolfram.jlink.MathJFrame"];
  (* setup the frame *)
  pane = frame@getContentPane[];
  pane@setLayout[JavaNew["java.awt.BorderLayout"]];
  mathCanvas=JavaNew["com.wolfram.jlink.MathGraphicsJPanel"];
  pane@add["Center", mathCanvas];
  (* draw image *)
  mathCanvas@setMathCommand[ToString[gr, InputForm]];
  img=mathCanvas@getImage[];
  (* add finishing touches *)
  {w, h} = img/@{getWidth[], getHeight[]};
  frame@setSize[w+30, h+60];
  frame@layout[];
  LoadClass["java.awt.Color"];
  frame@setBackground[Color`white];
  frame@setTitle["Mathematica Graphics: Out["<>ToString@$Line<>"]"];
  JavaShow[frame];
  gr
]]

(* this code remain in place for a direct call, but the old
   animations mechanism is removed in V6. *)

JavaAnimation[gr_List]:=
JavaBlock[Module[{fn, frame, exp},
  InstallJava[];
  (* create animated gif *)
  fn = Close@OpenWrite[]<>".gif";
  Export[fn, gr, ConversionOptions->{"Loop"->True}];
  (* display in window *)
  frame = JavaNew["com.wolfram.viewer.AnimateGIF", fn, $IconDirectory];
  (* add finishing touches *)
  frame@setTitle["Mathematica Animation: Out["<>ToString@$Line<>"]"];
  LoadClass["java.awt.Color"];
  frame@setBackground[Color`white];
  JavaShow[frame];
  gr
]]

(* sound *)
JavaSound[snd_] := 
JavaBlock[Module[{fn,frame},
  InstallJava[];
  (* create wav file *)
  fn = Close@OpenWrite[]<>".wav";
  Export[fn, snd];
  (* open player window *)
  frame = JavaNew["com.wolfram.viewer.PlaySound", fn, $IconDirectory];
  (* add finishing touches *)
  frame@setTitle["Mathematica Sound: Out["<>ToString@$Line<>"]"];
  LoadClass["java.awt.Color"];
  frame@setBackground[Color`white];
  JavaShow[frame];
  snd
]]

(* directory where GUI icons are stored *)
$IconDirectory=First@FileNames["JavaGraphics", $Path]

If[InstallJava[]=!=$Failed && Developer`InstallFrontEnd[]=!=$Failed,
  If[ !($BatchOutput || $Linked || $ParentLink =!= Null),
	Print[" -- Java graphics initialized -- "] ];
   $DisplayFunction = JavaDisplay;
   $SoundDisplayFunction = JavaSound,
  Message[Graphics::nogr, "Java"];
   $DisplayFunction = Identity;
   $SoundDisplayFunction = Identity
]

End[]

Null
