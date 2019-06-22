(* ::Package:: *)

{

(*
whenever there is a reliable way of detecting if NumLock is on, then take out this comment and start using the Alt+nnnn input method

"\[Euro]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad2Key]\\[NumPad8Key]\[RightModified]",

"\:201a" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad3Key]\\[NumPad0Key]\[RightModified]",

"\[Florin]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad3Key]\\[NumPad1Key]\[RightModified]",

"\:201e" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad3Key]\\[NumPad2Key]\[RightModified]",

"\[Ellipsis]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad3Key]\\[NumPad3Key]\[RightModified]",

"\[Dagger]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad3Key]\\[NumPad4Key]\[RightModified]",

"\[DoubleDagger]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad3Key]\\[NumPad5Key]\[RightModified]",

"\:02c6" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad3Key]\\[NumPad6Key]\[RightModified]",

"\:2030" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad3Key]\\[NumPad7Key]\[RightModified]",

"\[CapitalSHacek]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad3Key]\\[NumPad8Key]\[RightModified]",

"\:2039" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad3Key]\\[NumPad9Key]\[RightModified]",

"\:0152" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad4Key]\\[NumPad0Key]\[RightModified]",

"\:017d" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad4Key]\\[NumPad2Key]\[RightModified]",

"\[OpenCurlyQuote]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad4Key]\\[NumPad5Key]\[RightModified]",

"\[CloseCurlyQuote]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad4Key]\\[NumPad6Key]\[RightModified]",

"\[OpenCurlyDoubleQuote]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad4Key]\\[NumPad7Key]\[RightModified]",

"\[CloseCurlyDoubleQuote]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad4Key]\\[NumPad8Key]\[RightModified]",

"\[Bullet]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad4Key]\\[NumPad9Key]\[RightModified]",

"\[Dash]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad5Key]\\[NumPad0Key]\[RightModified]",

"\[LongDash]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad5Key]\\[NumPad1Key]\[RightModified]",

"\:02dc" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad5Key]\\[NumPad2Key]\[RightModified]",

"\[Trademark]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad5Key]\\[NumPad3Key]\[RightModified]",

"\[SHacek]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad5Key]\\[NumPad4Key]\[RightModified]",

"\:203a" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad5Key]\\[NumPad5Key]\[RightModified]",

"\:0153" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad5Key]\\[NumPad6Key]\[RightModified]",

"\:017e" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad5Key]\\[NumPad8Key]\[RightModified]",

"\:0178" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad5Key]\\[NumPad9Key]\[RightModified]",

"\[NonBreakingSpace]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad6Key]\\[NumPad0Key]\[RightModified]",

"\[DownExclamation]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad6Key]\\[NumPad1Key]\[RightModified]",

"\[Cent]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad6Key]\\[NumPad2Key]\[RightModified]",

"\[Sterling]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad6Key]\\[NumPad3Key]\[RightModified]",

"\[Currency]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad6Key]\\[NumPad4Key]\[RightModified]",

"\[Yen]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad6Key]\\[NumPad5Key]\[RightModified]",

"\246" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad6Key]\\[NumPad6Key]\[RightModified]",

"\[Section]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad6Key]\\[NumPad7Key]\[RightModified]",

"\250" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad6Key]\\[NumPad8Key]\[RightModified]",

"\[Copyright]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad6Key]\\[NumPad9Key]\[RightModified]",

"\252" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad7Key]\\[NumPad0Key]\[RightModified]",

"\[LeftGuillemet]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad7Key]\\[NumPad1Key]\[RightModified]",

"\[Not]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad7Key]\\[NumPad2Key]\[RightModified]",

"\[DiscretionaryHyphen]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad7Key]\\[NumPad3Key]\[RightModified]",

"\[RegisteredTrademark]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad7Key]\\[NumPad4Key]\[RightModified]",

"\257" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad7Key]\\[NumPad5Key]\[RightModified]",

"\[Degree]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad7Key]\\[NumPad6Key]\[RightModified]",

"\[PlusMinus]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad7Key]\\[NumPad7Key]\[RightModified]",

"\262" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad7Key]\\[NumPad8Key]\[RightModified]",

"\263" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad7Key]\\[NumPad9Key]\[RightModified]",

"\264" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad8Key]\\[NumPad0Key]\[RightModified]",

"\[Micro]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad8Key]\\[NumPad1Key]\[RightModified]",

"\[Paragraph]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad8Key]\\[NumPad2Key]\[RightModified]",

"\[CenterDot]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad8Key]\\[NumPad3Key]\[RightModified]",

"\270" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad8Key]\\[NumPad4Key]\[RightModified]",

"\271" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad8Key]\\[NumPad5Key]\[RightModified]",

"\272" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad8Key]\\[NumPad6Key]\[RightModified]",

"\[RightGuillemet]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad8Key]\\[NumPad7Key]\[RightModified]",

"\274" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad8Key]\\[NumPad8Key]\[RightModified]",

"\275" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad8Key]\\[NumPad9Key]\[RightModified]",

"\276" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad9Key]\\[NumPad0Key]\[RightModified]",

"\[DownQuestion]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad9Key]\\[NumPad1Key]\[RightModified]",

"\[CapitalAGrave]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad9Key]\\[NumPad2Key]\[RightModified]",

"\[CapitalAAcute]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad9Key]\\[NumPad3Key]\[RightModified]",

"\[CapitalAHat]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad9Key]\\[NumPad4Key]\[RightModified]",

"\[CapitalATilde]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad9Key]\\[NumPad5Key]\[RightModified]",

"\[CapitalADoubleDot]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad9Key]\\[NumPad6Key]\[RightModified]",

"\[CapitalARing]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad9Key]\\[NumPad7Key]\[RightModified]",

"\[CapitalAE]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad9Key]\\[NumPad8Key]\[RightModified]",

"\[CapitalCCedilla]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad1Key]\\[NumPad9Key]\\[NumPad9Key]\[RightModified]",

"\[CapitalEGrave]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad0Key]\\[NumPad0Key]\[RightModified]",

"\[CapitalEAcute]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad0Key]\\[NumPad1Key]\[RightModified]",

"\[CapitalEHat]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad0Key]\\[NumPad2Key]\[RightModified]",

"\[CapitalEDoubleDot]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad0Key]\\[NumPad3Key]\[RightModified]",

"\[CapitalIGrave]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad0Key]\\[NumPad4Key]\[RightModified]",

"\[CapitalIAcute]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad0Key]\\[NumPad5Key]\[RightModified]",

"\[CapitalIHat]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad0Key]\\[NumPad6Key]\[RightModified]",

"\[CapitalIDoubleDot]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad0Key]\\[NumPad7Key]\[RightModified]",

"\[CapitalEth]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad0Key]\\[NumPad8Key]\[RightModified]",

"\[CapitalNTilde]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad0Key]\\[NumPad9Key]\[RightModified]",

"\[CapitalOGrave]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad1Key]\\[NumPad0Key]\[RightModified]",

"\[CapitalOAcute]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad1Key]\\[NumPad1Key]\[RightModified]",

"\[CapitalOHat]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad1Key]\\[NumPad2Key]\[RightModified]",

"\[CapitalOTilde]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad1Key]\\[NumPad3Key]\[RightModified]",

"\[CapitalODoubleDot]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad1Key]\\[NumPad4Key]\[RightModified]",

"\[Times]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad1Key]\\[NumPad5Key]\[RightModified]",

"\[CapitalOSlash]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad1Key]\\[NumPad6Key]\[RightModified]",

"\[CapitalUGrave]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad1Key]\\[NumPad7Key]\[RightModified]",

"\[CapitalUAcute]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad1Key]\\[NumPad8Key]\[RightModified]",

"\[CapitalUHat]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad1Key]\\[NumPad9Key]\[RightModified]",

"\[CapitalUDoubleDot]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad2Key]\\[NumPad0Key]\[RightModified]",

"\[CapitalYAcute]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad2Key]\\[NumPad1Key]\[RightModified]",

"\[CapitalThorn]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad2Key]\\[NumPad2Key]\[RightModified]",

"\[SZ]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad2Key]\\[NumPad3Key]\[RightModified]",

"\[AGrave]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad2Key]\\[NumPad4Key]\[RightModified]",

"\[AAcute]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad2Key]\\[NumPad5Key]\[RightModified]",

"\[AHat]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad2Key]\\[NumPad6Key]\[RightModified]",

"\[ATilde]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad2Key]\\[NumPad7Key]\[RightModified]",

"\[ADoubleDot]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad2Key]\\[NumPad8Key]\[RightModified]",

"\[ARing]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad2Key]\\[NumPad9Key]\[RightModified]",

"\[AE]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad3Key]\\[NumPad0Key]\[RightModified]",

"\[CCedilla]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad3Key]\\[NumPad1Key]\[RightModified]",

"\[EGrave]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad3Key]\\[NumPad2Key]\[RightModified]",

"\[EAcute]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad3Key]\\[NumPad3Key]\[RightModified]",

"\[EHat]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad3Key]\\[NumPad4Key]\[RightModified]",

"\[EDoubleDot]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad3Key]\\[NumPad5Key]\[RightModified]",

"\[IGrave]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad3Key]\\[NumPad6Key]\[RightModified]",

"\[IAcute]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad3Key]\\[NumPad7Key]\[RightModified]",

"\[IHat]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad3Key]\\[NumPad8Key]\[RightModified]",

"\[IDoubleDot]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad3Key]\\[NumPad9Key]\[RightModified]",

"\[Eth]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad4Key]\\[NumPad0Key]\[RightModified]",

"\[NTilde]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad4Key]\\[NumPad1Key]\[RightModified]",

"\[OGrave]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad4Key]\\[NumPad2Key]\[RightModified]",

"\[OAcute]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad4Key]\\[NumPad3Key]\[RightModified]",

"\[OHat]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad4Key]\\[NumPad4Key]\[RightModified]",

"\[OTilde]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad4Key]\\[NumPad5Key]\[RightModified]",

"\[ODoubleDot]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad4Key]\\[NumPad6Key]\[RightModified]",

"\[Divide]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad4Key]\\[NumPad7Key]\[RightModified]",

"\[OSlash]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad4Key]\\[NumPad8Key]\[RightModified]",

"\[UGrave]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad4Key]\\[NumPad9Key]\[RightModified]",

"\[UAcute]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad5Key]\\[NumPad0Key]\[RightModified]",

"\[UHat]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad5Key]\\[NumPad1Key]\[RightModified]",

"\[UDoubleDot]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad5Key]\\[NumPad2Key]\[RightModified]",

"\[YAcute]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad5Key]\\[NumPad3Key]\[RightModified]",

"\[Thorn]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad5Key]\\[NumPad4Key]\[RightModified]",

"\[YDoubleDot]" -> "\[AltKey]\[LeftModified]\\[NumPad0Key]\\[NumPad2Key]\\[NumPad5Key]\\[NumPad5Key]\[RightModified]"
*)
}
