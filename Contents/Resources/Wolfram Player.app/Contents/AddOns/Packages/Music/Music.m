(* ::Package:: *)

(*:Mathematica Version: 2.0 *)

(*:Name: Music` *)

(*:Title: Music Functions *)

(*:Author: Arun Chandra (Wolfram Research), September 1992.  *)

(*:Summary:
This package provides functions for the manipulation and synthesis of scales.
It includes pitch/frequency equivalents.
*)

(*:Context: Music` *)

(*:Package Version: 1.0 *)

(* Updated September 2006 by Brian Van Vertloo - changed Scale[] to 
      MusicScale[] to prevent System` symbol overwrite. *)

(* :Copyright: Copyright 1992-2007, Wolfram Research, Inc.
*)

(*:Reference: Usage messages only. *)

(*:Keywords: sound, music, synthesis, composition *)

(*:Requirements: A system on which Mathematica can produce sound. *)

(*:Warning: None. *)

(*:Sources: 
    Brun, Herbert. 1991. My Words and Where I Want Them. London:
        Princelet Editions.
    Dodge, Charles. 1985. Computer Music.  New York: Schirmer Books.
    Hiller, Lejaren A. 1963-66. Lectures on Musical Acoustics. Unpublished.
    Mathews, Max V. 1969. The Technology of Computer Music. 
        Cambridge, MA: MIT Press.
    Moore, F. Richard. 1990. Elements of Computer Music. 
        Englewood Cliffs, NJ: Prentice-Hall.
    Olson, Harry F. 1967. Music, Physics, and Engineering. 
        New York: Dover Publications, Inc.
    Wells, Thomas H. 1981. The Technology of Electronic Music. 
        New York: Schirmer Books.
*)

BeginPackage["Music`"] ;

If[FileType[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"]]===File,
Select[FindList[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"],"Music`"],
StringMatchQ[#,StartOfString~~"Music`*"]&]//ToExpression;
];

If[Not@ValueQ[HertzToCents::usage],HertzToCents::usage =
"HertzToCents[flist] converts a list of frequencies measured in Hertz to a list \
of intervals measured in cents."];

If[Not@ValueQ[CentsToHertz::usage],CentsToHertz::usage =
"CentsToHertz[ilist] converts a list of intervals measured in cents to a \
list of frequencies measured in Hertz, beginning at 440 Hertz. \
CentsToHertz[ilist, f] gives a list of frequencies beginning at frequency f."];

If[Not@ValueQ[MusicScale::usage],MusicScale::usage =
"MusicScale[ilist, freq, dur] creates a Sound object corresponding to ilist, \
a list of intervals measured in cents, starting at freq Hertz and lasting \
dur seconds. Pre-defined interval lists are PythagoreanChromatic, \
PythagoreanMajor, JustMajor, JustMinor, MeanChromatic, MeanMajor, MeanMinor, \
TemperedChromatic, TemperedMajor, TemperedMinor, QuarterTone, and SixthTone."];

If[Not@ValueQ[PythagoreanChromatic::usage],PythagoreanChromatic::usage = "PythagoreanChromatic is an interval list \
that is an extension of the PythagoreanMajor scale. It has 21 \
notes, since the complete scale requires 7 \"natural\" notes, 7 \
\"flat\" notes, and 7 \"sharp\" notes."];

If[Not@ValueQ[PythagoreanMajor::usage],PythagoreanMajor::usage = "PythagoreanMajor is an interval list in \
which the members of the scale are derived from a sequence of \
octaves and fifths, where a fifth is the ratio of 3/2 (702 cents) \
and an octave is the ratio of 2/1 (1200 cents). The scale is built \
by successive fifth addition and octave subtraction."]; (* As
far as we know, PythagoreanMajor was invented in the 3rd or 4th
century, B.C.. *)

If[Not@ValueQ[JustMajor::usage],JustMajor::usage = "JustMajor is an interval list in which the ratios \
of the 3rd, 6th, and 7th intervals are simplified from the Pythagorean \
model. Whereas in the Pythagorean scale the ratios are 81/64 for a 3rd, \
27/16 for a 6th, and 243/128 for a 7th, in just intonation the ratios are \
5/4 for a 3rd, 5/3 for a 6th, and 15/8 for a 7th. The other intervals are \
the same as the Pythagorean scale. JustMajor was invented by the theorist \
Zarlino in the 16th century so that simultaneously sounding tones would have \
simple ratios."];

If[Not@ValueQ[JustMinor::usage],JustMinor::usage = "JustMinor is an interval list giving the minor \
version of the JustMajor scale."];

If[Not@ValueQ[MeanChromatic::usage],MeanChromatic::usage = "MeanChromatic is an interval list in which \
696.6 cents is used as the definition of a fifth, instead of 702 \
cents as in the Pythagorean and just intonation systems. This \
scale was invented in the 18th century by Gottfried Silbermann to \
correct for intonation problems due to enharmonic change."];

If[Not@ValueQ[MeanMajor::usage],MeanMajor::usage = "MeanMajor is an interval list derived from the \
MeanChromatic scale."];

If[Not@ValueQ[MeanMinor::usage],MeanMinor::usage = "MeanMinor is an interval list derived from the \
MeanChromatic scale."];

If[Not@ValueQ[TemperedChromatic::usage],TemperedChromatic::usage = "TemperedChromatic is an interval list \
corresponding to equal-temperament in which the octave is divided into 12 \
equal parts.  Each part is a tempered semitone (100 cents). This is  \
equivalent to making 12 fifths equal to 7 octaves, so an equal-tempered  \
fifth is equal to 700 cents. (The just intonation and Pythagorean fifths \
are 702 cents, and the Mean Tone fifth is 696.6 cents.) This process \
guarantees equivalence between pitches, and allows intervals to be \
the same in all keys. However, except for the octave, none of the \
intervals is in tune with regard to mathematical ratios and the \
logic Pythagoras developed from proportional lengths of strings."];

If[Not@ValueQ[TemperedMajor::usage],TemperedMajor::usage = "TemperedMajor is an interval list derived \
from the TemperedChromatic scale."];

If[Not@ValueQ[TemperedMinor::usage],TemperedMinor::usage = "TemperedMinor is an interval list derived \
from the TemperedChromatic scale."];

If[Not@ValueQ[QuarterTone::usage],QuarterTone::usage = "QuarterTone is an interval list in which each \
semitone (100 cents) is split in two."];

If[Not@ValueQ[SixthTone::usage],SixthTone::usage = "SixthTone is an interval list in which each \
semitone (100 cents) is split in three."];

(*Scan[(MessageName[Evaluate[ToExpression[#[[1]]<>#[[2]]]], "usage"] =
         #[[1]]<>#[[2]]<>" is the note "<>#[[1]]<>" in octave "<>#[[2]]<>".")&,
     Flatten[Outer[List,
     	{"A", "Asharp", "Bflat", "B", "Bsharp", "Cflat", "C", "Csharp", "Dflat",
	 "D", "Dsharp", "Eflat", "E", "Esharp", "Fflat", "F", "Fsharp", "Gflat",
         "G", "Gsharp", "Aflat"},
	{"0", "1", "2", "3", "4", "5", "6", "7"}], 1]]*)
If[Not@ValueQ[A0::usage],A0::usage = "A0 is the note A in octave 0."];
If[Not@ValueQ[A1::usage],A1::usage = "A1 is the note A in octave 1."];
If[Not@ValueQ[A2::usage],A2::usage = "A2 is the note A in octave 2."];
If[Not@ValueQ[A3::usage],A3::usage = "A3 is the note A in octave 3."];
If[Not@ValueQ[A4::usage],A4::usage = "A4 is the note A in octave 4."];
If[Not@ValueQ[A5::usage],A5::usage = "A5 is the note A in octave 5."];
If[Not@ValueQ[A6::usage],A6::usage = "A6 is the note A in octave 6."];
If[Not@ValueQ[A7::usage],A7::usage = "A7 is the note A in octave 7."];
If[Not@ValueQ[Asharp0::usage],Asharp0::usage = "Asharp0 is the note A-sharp in octave 0."];
If[Not@ValueQ[Asharp1::usage],Asharp1::usage = "Asharp1 is the note A-sharp in octave 1."];
If[Not@ValueQ[Asharp2::usage],Asharp2::usage = "Asharp2 is the note A-sharp in octave 2."];
If[Not@ValueQ[Asharp3::usage],Asharp3::usage = "Asharp3 is the note A-sharp in octave 3."];
If[Not@ValueQ[Asharp4::usage],Asharp4::usage = "Asharp4 is the note A-sharp in octave 4."];
If[Not@ValueQ[Asharp5::usage],Asharp5::usage = "Asharp5 is the note A-sharp in octave 5."];
If[Not@ValueQ[Asharp6::usage],Asharp6::usage = "Asharp6 is the note A-sharp in octave 6."];
If[Not@ValueQ[Asharp7::usage],Asharp7::usage = "Asharp7 is the note A-sharp in octave 7."];
If[Not@ValueQ[Bflat0::usage],Bflat0::usage = "Bflat0 is the note B-flat in octave 0."];
If[Not@ValueQ[Bflat1::usage],Bflat1::usage = "Bflat1 is the note B-flat in octave 1."];
If[Not@ValueQ[Bflat2::usage],Bflat2::usage = "Bflat2 is the note B-flat in octave 2."];
If[Not@ValueQ[Bflat3::usage],Bflat3::usage = "Bflat3 is the note B-flat in octave 3."];
If[Not@ValueQ[Bflat4::usage],Bflat4::usage = "Bflat4 is the note B-flat in octave 4."];
If[Not@ValueQ[Bflat5::usage],Bflat5::usage = "Bflat5 is the note B-flat in octave 5."];
If[Not@ValueQ[Bflat6::usage],Bflat6::usage = "Bflat6 is the note B-flat in octave 6."];
If[Not@ValueQ[Bflat7::usage],Bflat7::usage = "Bflat7 is the note B-flat in octave 7."];
If[Not@ValueQ[B0::usage],B0::usage = "B0 is the note B in octave 0."];
If[Not@ValueQ[B1::usage],B1::usage = "B1 is the note B in octave 1."];
If[Not@ValueQ[B2::usage],B2::usage = "B2 is the note B in octave 2."];
If[Not@ValueQ[B3::usage],B3::usage = "B3 is the note B in octave 3."];
If[Not@ValueQ[B4::usage],B4::usage = "B4 is the note B in octave 4."];
If[Not@ValueQ[B5::usage],B5::usage = "B5 is the note B in octave 5."];
If[Not@ValueQ[B6::usage],B6::usage = "B6 is the note B in octave 6."];
If[Not@ValueQ[B7::usage],B7::usage = "B7 is the note B in octave 7."];
If[Not@ValueQ[Bsharp0::usage],Bsharp0::usage = "Bsharp0 is the note B-sharp in octave 0."];
If[Not@ValueQ[Bsharp1::usage],Bsharp1::usage = "Bsharp1 is the note B-sharp in octave 1."];
If[Not@ValueQ[Bsharp2::usage],Bsharp2::usage = "Bsharp2 is the note B-sharp in octave 2."];
If[Not@ValueQ[Bsharp3::usage],Bsharp3::usage = "Bsharp3 is the note B-sharp in octave 3."];
If[Not@ValueQ[Bsharp4::usage],Bsharp4::usage = "Bsharp4 is the note B-sharp in octave 4."];
If[Not@ValueQ[Bsharp5::usage],Bsharp5::usage = "Bsharp5 is the note B-sharp in octave 5."];
If[Not@ValueQ[Bsharp6::usage],Bsharp6::usage = "Bsharp6 is the note B-sharp in octave 6."];
If[Not@ValueQ[Bsharp7::usage],Bsharp7::usage = "Bsharp7 is the note B-sharp in octave 7."];
If[Not@ValueQ[Cflat0::usage],Cflat0::usage = "Cflat0 is the note C-flat in octave 0."];
If[Not@ValueQ[Cflat1::usage],Cflat1::usage = "Cflat1 is the note C-flat in octave 1."];
If[Not@ValueQ[Cflat2::usage],Cflat2::usage = "Cflat2 is the note C-flat in octave 2."];
If[Not@ValueQ[Cflat3::usage],Cflat3::usage = "Cflat3 is the note C-flat in octave 3."];
If[Not@ValueQ[Cflat4::usage],Cflat4::usage = "Cflat4 is the note C-flat in octave 4."];
If[Not@ValueQ[Cflat5::usage],Cflat5::usage = "Cflat5 is the note C-flat in octave 5."];
If[Not@ValueQ[Cflat6::usage],Cflat6::usage = "Cflat6 is the note C-flat in octave 6."];
If[Not@ValueQ[Cflat7::usage],Cflat7::usage = "Cflat7 is the note C-flat in octave 7."];
If[Not@ValueQ[C0::usage],C0::usage = "C0 is the note C in octave 0."];
If[Not@ValueQ[C1::usage],C1::usage = "C1 is the note C in octave 1."];
If[Not@ValueQ[C2::usage],C2::usage = "C2 is the note C in octave 2."];
If[Not@ValueQ[C3::usage],C3::usage = "C3 is the note C in octave 3."];
If[Not@ValueQ[C4::usage],C4::usage = "C4 is the note C in octave 4."];
If[Not@ValueQ[C5::usage],C5::usage = "C5 is the note C in octave 5."];
If[Not@ValueQ[C6::usage],C6::usage = "C6 is the note C in octave 6."];
If[Not@ValueQ[C7::usage],C7::usage = "C7 is the note C in octave 7."];
If[Not@ValueQ[Csharp0::usage],Csharp0::usage = "Csharp0 is the note C-sharp in octave 0."];
If[Not@ValueQ[Csharp1::usage],Csharp1::usage = "Csharp1 is the note C-sharp in octave 1."];
If[Not@ValueQ[Csharp2::usage],Csharp2::usage = "Csharp2 is the note C-sharp in octave 2."];
If[Not@ValueQ[Csharp3::usage],Csharp3::usage = "Csharp3 is the note C-sharp in octave 3."];
If[Not@ValueQ[Csharp4::usage],Csharp4::usage = "Csharp4 is the note C-sharp in octave 4."];
If[Not@ValueQ[Csharp5::usage],Csharp5::usage = "Csharp5 is the note C-sharp in octave 5."];
If[Not@ValueQ[Csharp6::usage],Csharp6::usage = "Csharp6 is the note C-sharp in octave 6."];
If[Not@ValueQ[Csharp7::usage],Csharp7::usage = "Csharp7 is the note C-sharp in octave 7."];
If[Not@ValueQ[Dflat0::usage],Dflat0::usage = "Dflat0 is the note D-flat in octave 0."];
If[Not@ValueQ[Dflat1::usage],Dflat1::usage = "Dflat1 is the note D-flat in octave 1."];
If[Not@ValueQ[Dflat2::usage],Dflat2::usage = "Dflat2 is the note D-flat in octave 2."];
If[Not@ValueQ[Dflat3::usage],Dflat3::usage = "Dflat3 is the note D-flat in octave 3."];
If[Not@ValueQ[Dflat4::usage],Dflat4::usage = "Dflat4 is the note D-flat in octave 4."];
If[Not@ValueQ[Dflat5::usage],Dflat5::usage = "Dflat5 is the note D-flat in octave 5."];
If[Not@ValueQ[Dflat6::usage],Dflat6::usage = "Dflat6 is the note D-flat in octave 6."];
If[Not@ValueQ[Dflat7::usage],Dflat7::usage = "Dflat7 is the note D-flat in octave 7."];
If[Not@ValueQ[D0::usage],D0::usage = "D0 is the note D in octave 0."];
If[Not@ValueQ[D1::usage],D1::usage = "D1 is the note D in octave 1."];
If[Not@ValueQ[D2::usage],D2::usage = "D2 is the note D in octave 2."];
If[Not@ValueQ[D3::usage],D3::usage = "D3 is the note D in octave 3."];
If[Not@ValueQ[D4::usage],D4::usage = "D4 is the note D in octave 4."];
If[Not@ValueQ[D5::usage],D5::usage = "D5 is the note D in octave 5."];
If[Not@ValueQ[D6::usage],D6::usage = "D6 is the note D in octave 6."];
If[Not@ValueQ[D7::usage],D7::usage = "D7 is the note D in octave 7."];
If[Not@ValueQ[Dsharp0::usage],Dsharp0::usage = "Dsharp0 is the note D-sharp in octave 0."];
If[Not@ValueQ[Dsharp1::usage],Dsharp1::usage = "Dsharp1 is the note D-sharp in octave 1."];
If[Not@ValueQ[Dsharp2::usage],Dsharp2::usage = "Dsharp2 is the note D-sharp in octave 2."];
If[Not@ValueQ[Dsharp3::usage],Dsharp3::usage = "Dsharp3 is the note D-sharp in octave 3."];
If[Not@ValueQ[Dsharp4::usage],Dsharp4::usage = "Dsharp4 is the note D-sharp in octave 4."];
If[Not@ValueQ[Dsharp5::usage],Dsharp5::usage = "Dsharp5 is the note D-sharp in octave 5."];
If[Not@ValueQ[Dsharp6::usage],Dsharp6::usage = "Dsharp6 is the note D-sharp in octave 6."];
If[Not@ValueQ[Dsharp7::usage],Dsharp7::usage = "Dsharp7 is the note D-sharp in octave 7."];
If[Not@ValueQ[Eflat0::usage],Eflat0::usage = "Eflat0 is the note E-flat in octave 0."];
If[Not@ValueQ[Eflat1::usage],Eflat1::usage = "Eflat1 is the note E-flat in octave 1."];
If[Not@ValueQ[Eflat2::usage],Eflat2::usage = "Eflat2 is the note E-flat in octave 2."];
If[Not@ValueQ[Eflat3::usage],Eflat3::usage = "Eflat3 is the note E-flat in octave 3."];
If[Not@ValueQ[Eflat4::usage],Eflat4::usage = "Eflat4 is the note E-flat in octave 4."];
If[Not@ValueQ[Eflat5::usage],Eflat5::usage = "Eflat5 is the note E-flat in octave 5."];
If[Not@ValueQ[Eflat6::usage],Eflat6::usage = "Eflat6 is the note E-flat in octave 6."];
If[Not@ValueQ[Eflat7::usage],Eflat7::usage = "Eflat7 is the note E-flat in octave 7."];
If[Not@ValueQ[E0::usage],E0::usage = "E0 is the note E in octave 0."];
If[Not@ValueQ[E1::usage],E1::usage = "E1 is the note E in octave 1."];
If[Not@ValueQ[E2::usage],E2::usage = "E2 is the note E in octave 2."];
If[Not@ValueQ[E3::usage],E3::usage = "E3 is the note E in octave 3."];
If[Not@ValueQ[E4::usage],E4::usage = "E4 is the note E in octave 4."];
If[Not@ValueQ[E5::usage],E5::usage = "E5 is the note E in octave 5."];
If[Not@ValueQ[E6::usage],E6::usage = "E6 is the note E in octave 6."];
If[Not@ValueQ[E7::usage],E7::usage = "E7 is the note E in octave 7."];
If[Not@ValueQ[Esharp0::usage],Esharp0::usage = "Esharp0 is the note E-sharp in octave 0."];
If[Not@ValueQ[Esharp1::usage],Esharp1::usage = "Esharp1 is the note E-sharp in octave 1."];
If[Not@ValueQ[Esharp2::usage],Esharp2::usage = "Esharp2 is the note E-sharp in octave 2."];
If[Not@ValueQ[Esharp3::usage],Esharp3::usage = "Esharp3 is the note E-sharp in octave 3."];
If[Not@ValueQ[Esharp4::usage],Esharp4::usage = "Esharp4 is the note E-sharp in octave 4."];
If[Not@ValueQ[Esharp5::usage],Esharp5::usage = "Esharp5 is the note E-sharp in octave 5."];
If[Not@ValueQ[Esharp6::usage],Esharp6::usage = "Esharp6 is the note E-sharp in octave 6."];
If[Not@ValueQ[Esharp7::usage],Esharp7::usage = "Esharp7 is the note E-sharp in octave 7."];
If[Not@ValueQ[Fflat0::usage],Fflat0::usage = "Fflat0 is the note F-flat in octave 0."];
If[Not@ValueQ[Fflat1::usage],Fflat1::usage = "Fflat1 is the note F-flat in octave 1."];
If[Not@ValueQ[Fflat2::usage],Fflat2::usage = "Fflat2 is the note F-flat in octave 2."];
If[Not@ValueQ[Fflat3::usage],Fflat3::usage = "Fflat3 is the note F-flat in octave 3."];
If[Not@ValueQ[Fflat4::usage],Fflat4::usage = "Fflat4 is the note F-flat in octave 4."];
If[Not@ValueQ[Fflat5::usage],Fflat5::usage = "Fflat5 is the note F-flat in octave 5."];
If[Not@ValueQ[Fflat6::usage],Fflat6::usage = "Fflat6 is the note F-flat in octave 6."];
If[Not@ValueQ[Fflat7::usage],Fflat7::usage = "Fflat7 is the note F-flat in octave 7."];
If[Not@ValueQ[F0::usage],F0::usage = "F0 is the note F in octave 0."];
If[Not@ValueQ[F1::usage],F1::usage = "F1 is the note F in octave 1."];
If[Not@ValueQ[F2::usage],F2::usage = "F2 is the note F in octave 2."];
If[Not@ValueQ[F3::usage],F3::usage = "F3 is the note F in octave 3."];
If[Not@ValueQ[F4::usage],F4::usage = "F4 is the note F in octave 4."];
If[Not@ValueQ[F5::usage],F5::usage = "F5 is the note F in octave 5."];
If[Not@ValueQ[F6::usage],F6::usage = "F6 is the note F in octave 6."];
If[Not@ValueQ[F7::usage],F7::usage = "F7 is the note F in octave 7."];
If[Not@ValueQ[Fsharp0::usage],Fsharp0::usage = "Fsharp0 is the note F-sharp in octave 0."];
If[Not@ValueQ[Fsharp1::usage],Fsharp1::usage = "Fsharp1 is the note F-sharp in octave 1."];
If[Not@ValueQ[Fsharp2::usage],Fsharp2::usage = "Fsharp2 is the note F-sharp in octave 2."];
If[Not@ValueQ[Fsharp3::usage],Fsharp3::usage = "Fsharp3 is the note F-sharp in octave 3."];
If[Not@ValueQ[Fsharp4::usage],Fsharp4::usage = "Fsharp4 is the note F-sharp in octave 4."];
If[Not@ValueQ[Fsharp5::usage],Fsharp5::usage = "Fsharp5 is the note F-sharp in octave 5."];
If[Not@ValueQ[Fsharp6::usage],Fsharp6::usage = "Fsharp6 is the note F-sharp in octave 6."];
If[Not@ValueQ[Fsharp7::usage],Fsharp7::usage = "Fsharp7 is the note F-sharp in octave 7."];
If[Not@ValueQ[Gflat0::usage],Gflat0::usage = "Gflat0 is the note G-flat in octave 0."];
If[Not@ValueQ[Gflat1::usage],Gflat1::usage = "Gflat1 is the note G-flat in octave 1."];
If[Not@ValueQ[Gflat2::usage],Gflat2::usage = "Gflat2 is the note G-flat in octave 2."];
If[Not@ValueQ[Gflat3::usage],Gflat3::usage = "Gflat3 is the note G-flat in octave 3."];
If[Not@ValueQ[Gflat4::usage],Gflat4::usage = "Gflat4 is the note G-flat in octave 4."];
If[Not@ValueQ[Gflat5::usage],Gflat5::usage = "Gflat5 is the note G-flat in octave 5."];
If[Not@ValueQ[Gflat6::usage],Gflat6::usage = "Gflat6 is the note G-flat in octave 6."];
If[Not@ValueQ[Gflat7::usage],Gflat7::usage = "Gflat7 is the note G-flat in octave 7."];
If[Not@ValueQ[G0::usage],G0::usage = "G0 is the note G in octave 0."];
If[Not@ValueQ[G1::usage],G1::usage = "G1 is the note G in octave 1."];
If[Not@ValueQ[G2::usage],G2::usage = "G2 is the note G in octave 2."];
If[Not@ValueQ[G3::usage],G3::usage = "G3 is the note G in octave 3."];
If[Not@ValueQ[G4::usage],G4::usage = "G4 is the note G in octave 4."];
If[Not@ValueQ[G5::usage],G5::usage = "G5 is the note G in octave 5."];
If[Not@ValueQ[G6::usage],G6::usage = "G6 is the note G in octave 6."];
If[Not@ValueQ[G7::usage],G7::usage = "G7 is the note G in octave 7."];
If[Not@ValueQ[Gsharp0::usage],Gsharp0::usage = "Gsharp0 is the note G-sharp in octave 0."];
If[Not@ValueQ[Gsharp1::usage],Gsharp1::usage = "Gsharp1 is the note G-sharp in octave 1."];
If[Not@ValueQ[Gsharp2::usage],Gsharp2::usage = "Gsharp2 is the note G-sharp in octave 2."];
If[Not@ValueQ[Gsharp3::usage],Gsharp3::usage = "Gsharp3 is the note G-sharp in octave 3."];
If[Not@ValueQ[Gsharp4::usage],Gsharp4::usage = "Gsharp4 is the note G-sharp in octave 4."];
If[Not@ValueQ[Gsharp5::usage],Gsharp5::usage = "Gsharp5 is the note G-sharp in octave 5."];
If[Not@ValueQ[Gsharp6::usage],Gsharp6::usage = "Gsharp6 is the note G-sharp in octave 6."];
If[Not@ValueQ[Gsharp7::usage],Gsharp7::usage = "Gsharp7 is the note G-sharp in octave 7."];
If[Not@ValueQ[Aflat0::usage],Aflat0::usage = "Aflat0 is the note A-flat in octave 0."];
If[Not@ValueQ[Aflat1::usage],Aflat1::usage = "Aflat1 is the note A-flat in octave 1."];
If[Not@ValueQ[Aflat2::usage],Aflat2::usage = "Aflat2 is the note A-flat in octave 2."];
If[Not@ValueQ[Aflat3::usage],Aflat3::usage = "Aflat3 is the note A-flat in octave 3."];
If[Not@ValueQ[Aflat4::usage],Aflat4::usage = "Aflat4 is the note A-flat in octave 4."];
If[Not@ValueQ[Aflat5::usage],Aflat5::usage = "Aflat5 is the note A-flat in octave 5."];
If[Not@ValueQ[Aflat6::usage],Aflat6::usage = "Aflat6 is the note A-flat in octave 6."];
If[Not@ValueQ[Aflat7::usage],Aflat7::usage = "Aflat7 is the note A-flat in octave 7."];


Begin["`Private`"] ;

Unprotect[MusicScale, PythagoreanChromatic, PythagoreanMajor, JustMajor, JustMinor,
    MeanChromatic, MeanMajor, MeanMinor, TemperedChromatic, TemperedMajor,
    TemperedMinor, QuarterTone, SixthTone, HertzToCents, CentsToHertz];


(*

    List of defined pitches

*)

(* The package has had the following definition of pitch names since it
   was written. These do not, unfortunately, correspond with standard
   notation. My guess is that this matched a MIDI form current when Arun
   wrote the package, but it is clearly wrong for virtually all current
   applications. Because it has been around for so long, however, we
   want to provide a compatibility mechanism for users to get these
   values back. As such, I'm providing two undocumented (outside of here)
   methods: a variable that can be set in a user's Kernel/init.m, and
   a function that can be run after package load. For the init, evaluate:
   Music`$UseOldPitches = True
   before loading Music`. To set after loading the package, evaluate:
   Music`GenerateOldPitches[]
   --JMN 130714
*)
Music`GenerateOldPitches[] :=
If [ ! NumberQ[A1],
	notes = {"A", "Asharp", "B", "C", "Csharp", "D", "Dsharp", 
		"E", "F", "Fsharp", "G", "Gsharp"};
	Do[Evaluate[ToExpression[notes[[Mod[k-1,12]+1]] <>
			ToString[Ceiling[k/12]-1]]] = 27.5 2.^((k-1)/12), {k, 96}];
	Do[oct = ToString[k];
		Evaluate[ToExpression["Bsharp" <> oct]] = ToExpression["C" <> oct];
		Evaluate[ToExpression["Esharp" <> oct]] = ToExpression["F" <> oct];
		Evaluate[ToExpression["Cflat" <> oct]] = ToExpression["B" <> oct];
		Evaluate[ToExpression["Fflat" <> oct]] = ToExpression["E" <> oct];
		Evaluate[ToExpression["Bflat" <> oct]] = ToExpression["Asharp" <> oct];
		Evaluate[ToExpression["Dflat" <> oct]] = ToExpression["Csharp" <> oct];
		Evaluate[ToExpression["Eflat" <> oct]] = ToExpression["Dsharp" <> oct];
		Evaluate[ToExpression["Gflat" <> oct]] = ToExpression["Fsharp" <> oct];
		Evaluate[ToExpression["Aflat" <> oct]] = ToExpression["Gsharp" <> oct],
		{k, 0, 7}]
]
If[Music`$UseOldPitches === True, Music`GenerateOldPitches[]];

(* These are the correct namings, based on A4 = 440Hz *)
(* I'm sure this can be done more elegantly; this is modeled after the above. *)
If[!NumberQ[C1],
notes = {"C", "Csharp", "D", "Dsharp", "E", "F", "Fsharp", "G", "Gsharp", "A",
         "Asharp", "B"};
Do[Evaluate[Symbol[notes[[Mod[k - 3, 12] + 1]] <> 
        ToString[Ceiling[(k - 2)/12] + 4]]] =  440 2.^(k/12),
   {k, -57, 38}];
Do[oct = ToString[k];
    If[k < 7,
        Evaluate[Symbol["Bsharp" <> oct]] = Symbol["C" <> ToString[k + 1]],
        Bsharp7 = 440 2.^(39/12)
    ];
    Evaluate[Symbol["Esharp" <> oct]] = Symbol["F" <> oct];
    If[k > 0,
        Evaluate[Symbol["Cflat" <> oct]] = Symbol["B" <> ToString[k - 1]],
        Cflat0 = 440 2.^(-58/12)
    ];
    Evaluate[Symbol["Fflat" <> oct]] = Symbol["E" <> oct];
    Evaluate[Symbol["Bflat" <> oct]] = Symbol["Asharp" <> oct];
    Evaluate[Symbol["Dflat" <> oct]] = Symbol["Csharp" <> oct];
    Evaluate[Symbol["Eflat" <> oct]] = Symbol["Dsharp" <> oct];
    Evaluate[Symbol["Gflat" <> oct]] = Symbol["Fsharp" <> oct];
    Evaluate[Symbol["Aflat" <> oct]] = Symbol["Gsharp" <> oct],
   {k, 0, 7}
]
];
   

(*

    Set the default values for SampleRate, SampleDepth, and PlayRange.

*)

{sr, sd} = Switch[ $System,
        "NeXT", {22050, 16},
        "SPARC", {8000, 8},
        "Macintosh", {22254.5454, 8},
        "386", {11025, 8},
        "486", {11025, 8},
        _, {8192, 8}
];

Options[MusicScale] = { SampleRate -> sr, SampleDepth -> sd, 
	PlayRange -> {-1,1}, DisplayFunction->Identity};

(*

    Scale: All the following scales represent their intervals in
    cents, where 1200 cents == 1 octave.

*)

PythagoreanChromatic = {0,24,90,114,204,294,318,384,408,498,
    522,588,612,702,798,816,906,996,1020,1086,1110,1200};
PythagoreanMajor = {0,204,408,498,702,906,1110,1200};

JustMajor = {0,204,386.4,498,702,884.4,1088.4,1200};
JustMinor = {0,204,315.6,498,702,813.7,996.1,1200};

MeanChromatic = {0,76,193.2,310.3,386.3,503.4,579.5,
                696.6,772.6,889.7,1006.8,1082.9,1200};
MeanMajor = {0,193.2,386.3,503.4,696.6,889.7,1082.9,1200};
MeanMinor = {0,193.2,310.3,503.4,696.6,772.6,1006.8,1200};

TemperedChromatic = {0,100,200,300,400,500,600,700, 800,900,1000,1100,1200};
TemperedMajor = {0,200,400,500,700,900,1100,1200};
TemperedMinor = {0,200,300,500,700,800,1000,1200};

QuarterTone = {0,50,100,150,200,250,300,350,400,450,500,550,
                600,650,700,750,800,850,900,950,1000,1050,
                1100,1150,1200};

SixthTone = {0,33,66,100,133,166,200,233,266,300,333,366,
            400,433,466,500,533,566,600,633,666,700,733,
            766,800,833,866,900,933,966,1000,1033,1066,1100,
            1133,1166,1200};

isNumList[zlist_] := Return[ Apply[And, Map[NumberQ, zlist]] ] ;
Music::notnums = "Some members of the list `1` are not numbers.";

MusicScale::tooshort = "Interval list `` must have at least two members.";

MusicScale[ 
	i_?((VectorQ[#, NumberQ[N[#]]&])&),
	f_?((NumberQ[#])&), d_?((NumberQ[#])&), opts___] := 
		With[ { out = scale[i, f, d, opts] }, 
			out /; out =!= $Failed ] /; Length[i] >= 2
	

scale[ Intervals_, startingFreq_, totalDuration_, opts___ ] :=

  Module[
    {intervalList, noteDuration, sr, sd, pr, id, mypi},

	{ sr, sd, pr, id } =
        {SampleRate, SampleDepth, PlayRange, DisplayFunction} 
			/. {opts} /. Options[MusicScale];

    intervalList = Map[(N[startingFreq * 10^(#/(1200/Log[10,2]))])&, Intervals];
    noteDuration = Length[intervalList]/totalDuration;
	mypi = N[2 Pi] ;

    Play[ Sin[ mypi t intervalList[[ 1+Floor[t * noteDuration] ]] ],
          {t,0,totalDuration-.000001},
		SampleRate->sr, SampleDepth->sd, PlayRange->pr, DisplayFunction->id ]
]

h2c[x_, y_] := N[ 3986.313714 * ( Log[10, y] - Log[10, x] ) ] ;

HertzToCents[hlist_?((VectorQ[#, NumberQ[N[#]]&])&)] := 
	Apply[(h2c[#1,#2])&, Partition[hlist, 2, 1],{1}] /; Length[hlist] >= 2


CentsToHertz[clist_?((VectorQ[#, NumberQ[N[#]]&])&), f_:440] := 
	Map[(N[f * 10^(#/(1200/Log[10,2]))])&, clist] /; 
					Length[clist] >= 2 && NumberQ[N[f]]


(*

        Protect all user-accessible functions.

*)



End[] ;
Protect[MusicScale, PythagoreanChromatic, PythagoreanMajor, JustMajor, JustMinor,
    MeanChromatic, MeanMajor, MeanMinor, TemperedChromatic, TemperedMajor,
    TemperedMinor, QuarterTone, SixthTone, HertzToCents, CentsToHertz];
    
EndPackage[] ;



