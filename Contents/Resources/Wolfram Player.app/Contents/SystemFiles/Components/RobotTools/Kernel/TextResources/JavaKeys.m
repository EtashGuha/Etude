(* ::Package:: *)

(*
these are fakes that must come before the "\\" -> KeyEvent`VKUBACKUSLASH rule,
or else the pattern matcher breaks apart the fake key
*)

{
(*http://en.wikipedia.org/wiki/AltGr_key*)
"\\[AltGrKey]" -> java`awt`event`KeyEvent`VKUALTUGRAPH,

"\\[BackspaceKey]" -> java`awt`event`KeyEvent`VKUBACKUSPACE,

"\\[CapsLockKey]" -> java`awt`event`KeyEvent`VKUCAPSULOCK,

"\\[DownKey]" -> java`awt`event`KeyEvent`VKUDOWN,

"\\[EuroSignKey]" -> java`awt`event`KeyEvent`VKUEUROSIGN,

"\\[EndKey]" -> java`awt`event`KeyEvent`VKUEND,

"\\[F1Key]" -> java`awt`event`KeyEvent`VKUF1,

"\\[F2Key]" -> java`awt`event`KeyEvent`VKUF2,

"\\[F3Key]" -> java`awt`event`KeyEvent`VKUF3,

"\\[F4Key]" -> java`awt`event`KeyEvent`VKUF4,

"\\[F5Key]" -> java`awt`event`KeyEvent`VKUF5,

"\\[F6Key]" -> java`awt`event`KeyEvent`VKUF6,

"\\[F7Key]" -> java`awt`event`KeyEvent`VKUF7,

"\\[F8Key]" -> java`awt`event`KeyEvent`VKUF8,

"\\[F9Key]" -> java`awt`event`KeyEvent`VKUF9,

"\\[F10Key]" -> java`awt`event`KeyEvent`VKUF10,

"\\[F11Key]" -> java`awt`event`KeyEvent`VKUF11,

"\\[F12Key]" -> java`awt`event`KeyEvent`VKUF12,

"\\[F13Key]" -> java`awt`event`KeyEvent`VKUF13,

"\\[F14Key]" -> java`awt`event`KeyEvent`VKUF14,

"\\[F15Key]" -> java`awt`event`KeyEvent`VKUF15,

"\\[F16Key]" -> java`awt`event`KeyEvent`VKUF16,

"\\[F17Key]" -> java`awt`event`KeyEvent`VKUF17,

"\\[F18Key]" -> java`awt`event`KeyEvent`VKUF18,

"\\[F19Key]" -> java`awt`event`KeyEvent`VKUF19,

"\\[F20Key]" -> java`awt`event`KeyEvent`VKUF20,

"\\[F21Key]" -> java`awt`event`KeyEvent`VKUF21,

"\\[F22Key]" -> java`awt`event`KeyEvent`VKUF22,

"\\[F23Key]" -> java`awt`event`KeyEvent`VKUF23,

"\\[F24Key]" -> java`awt`event`KeyEvent`VKUF24,

"\\[HelpKey]" -> java`awt`event`KeyEvent`VKUHELP,

"\\[HiraganaKey]" -> java`awt`event`KeyEvent`VKUHIRAGANA,

"\\[HomeKey]" -> java`awt`event`KeyEvent`VKUHOME,

"\\[InsertKey]" -> java`awt`event`KeyEvent`VKUINSERT,

"\\[JapaneseHiraganaKey]" -> java`awt`event`KeyEvent`VKUJAPANESEUHIRAGANA,

"\\[JapaneseKatakanaKey]" -> java`awt`event`KeyEvent`VKUJAPANESEUKATAKANA,

"\\[JapaneseRomanKey]" -> java`awt`event`KeyEvent`VKUJAPANESEUROMAN,

"\\[KanaKey]" -> java`awt`event`KeyEvent`VKUKANA,

"\\[KanaLockKey]" -> java`awt`event`KeyEvent`VKUKANAULOCK,

"\\[KanjiKey]" -> java`awt`event`KeyEvent`VKUKANJI,

"\\[KatakanaKey]" -> java`awt`event`KeyEvent`VKUKATAKANA,

"\\[KeyPadDownKey]" -> java`awt`event`KeyEvent`VKUKPUDOWN,

"\\[KeyPadLeftKey]" -> java`awt`event`KeyEvent`VKUKPULEFT,

"\\[KeyPadRightKey]" -> java`awt`event`KeyEvent`VKUKPURIGHT,

"\\[KeyPadUpKey]" -> java`awt`event`KeyEvent`VKUKPUUP,

"\\[LeftKey]" -> java`awt`event`KeyEvent`VKULEFT,

(*http://en.wikipedia.org/wiki/Meta_key*)
"\\[MetaKey]" -> java`awt`event`KeyEvent`VKUMETA,

"\\[NumLockKey]" -> java`awt`event`KeyEvent`VKUNUMULOCK,

"\\[NumPad0Key]" -> java`awt`event`KeyEvent`VKUNUMPAD0,

"\\[NumPad1Key]" -> java`awt`event`KeyEvent`VKUNUMPAD1,

"\\[NumPad2Key]" -> java`awt`event`KeyEvent`VKUNUMPAD2,

"\\[NumPad3Key]" -> java`awt`event`KeyEvent`VKUNUMPAD3,

"\\[NumPad4Key]" -> java`awt`event`KeyEvent`VKUNUMPAD4,

"\\[NumPad5Key]" -> java`awt`event`KeyEvent`VKUNUMPAD5,

"\\[NumPad6Key]" -> java`awt`event`KeyEvent`VKUNUMPAD6,

"\\[NumPad7Key]" -> java`awt`event`KeyEvent`VKUNUMPAD7,

"\\[NumPad8Key]" -> java`awt`event`KeyEvent`VKUNUMPAD8,

"\\[NumPad9Key]" -> java`awt`event`KeyEvent`VKUNUMPAD9,

"\\[PageDownKey]" -> java`awt`event`KeyEvent`VKUPAGEUDOWN,

"\\[PageUpKey]" -> java`awt`event`KeyEvent`VKUPAGEUUP,

"\\[PauseKey]" -> java`awt`event`KeyEvent`VKUPAUSE,

"\\[PrintScreenKey]" -> java`awt`event`KeyEvent`VKUPRINTSCREEN,

"\\[RightKey]" -> java`awt`event`KeyEvent`VKURIGHT,

"\\[RomanCharactersKey]" -> java`awt`event`KeyEvent`VKUROMANUCHARACTERS,

"\\[ScrollLockKey]" -> java`awt`event`KeyEvent`VKUSCROLLULOCK,

"\\[UpKey]" -> java`awt`event`KeyEvent`VKUUP,

"\\[WindowsKey]" -> java`awt`event`KeyEvent`VKUWINDOWS,

(* these are real *)
"0" -> java`awt`event`KeyEvent`VKU0,

"1" -> java`awt`event`KeyEvent`VKU1,

"2" -> java`awt`event`KeyEvent`VKU2,

"3" -> java`awt`event`KeyEvent`VKU3,

"4" -> java`awt`event`KeyEvent`VKU4,

"5" -> java`awt`event`KeyEvent`VKU5,

"6" -> java`awt`event`KeyEvent`VKU6,

"7" -> java`awt`event`KeyEvent`VKU7,

"8" -> java`awt`event`KeyEvent`VKU8,

"9" -> java`awt`event`KeyEvent`VKU9,

"a" -> java`awt`event`KeyEvent`VKUA,

"b" -> java`awt`event`KeyEvent`VKUB,

"c" -> java`awt`event`KeyEvent`VKUC,

"d" -> java`awt`event`KeyEvent`VKUD,

"e" -> java`awt`event`KeyEvent`VKUE,

"f" -> java`awt`event`KeyEvent`VKUF,

"g" -> java`awt`event`KeyEvent`VKUG,

"h" -> java`awt`event`KeyEvent`VKUH,

"i" -> java`awt`event`KeyEvent`VKUI,

"j" -> java`awt`event`KeyEvent`VKUJ,

"k" -> java`awt`event`KeyEvent`VKUK,

"l" -> java`awt`event`KeyEvent`VKUL,

"m" -> java`awt`event`KeyEvent`VKUM,

"n" -> java`awt`event`KeyEvent`VKUN,

"o" -> java`awt`event`KeyEvent`VKUO,

"p" -> java`awt`event`KeyEvent`VKUP,

"q" -> java`awt`event`KeyEvent`VKUQ,

"r" -> java`awt`event`KeyEvent`VKUR,

"s" -> java`awt`event`KeyEvent`VKUS,

"t" -> java`awt`event`KeyEvent`VKUT,

"u" -> java`awt`event`KeyEvent`VKUU,

"v" -> java`awt`event`KeyEvent`VKUV,

"w" -> java`awt`event`KeyEvent`VKUW,

"x" -> java`awt`event`KeyEvent`VKUX,

"y" -> java`awt`event`KeyEvent`VKUY,

"z" -> java`awt`event`KeyEvent`VKUZ,

"\[AltKey]" -> java`awt`event`KeyEvent`VKUALT,

"\[OptionKey]" -> java`awt`event`KeyEvent`VKUALT,

"`" -> java`awt`event`KeyEvent`VKUBACKUQUOTE,

"\\" -> java`awt`event`KeyEvent`VKUBACKUSLASH,

"]" -> java`awt`event`KeyEvent`VKUCLOSEUBRACKET,

"," -> java`awt`event`KeyEvent`VKUCOMMA,

"\[ControlKey]" -> java`awt`event`KeyEvent`VKUCONTROL,

"\[DeleteKey]" -> java`awt`event`KeyEvent`VKUDELETE,

"$" -> java`awt`event`KeyEvent`VKUDOLLAR,

"\[EnterKey]" -> java`awt`event`KeyEvent`VKUENTER,

"\[ReturnKey]" -> java`awt`event`KeyEvent`VKUENTER,

"\[SystemEnterKey]" -> java`awt`event`KeyEvent`VKUENTER,

"=" -> java`awt`event`KeyEvent`VKUEQUALS,

"\[EscapeKey]" -> java`awt`event`KeyEvent`VKUESCAPE,

"\[DownExclamation]" -> java`awt`event`KeyEvent`VKUINVERTEDUEXCLAMATIONUMARK,

"\[CommandKey]" -> java`awt`event`KeyEvent`VKUMETA,

"-" -> java`awt`event`KeyEvent`VKUMINUS,

"[" -> java`awt`event`KeyEvent`VKUOPENUBRACKET,

"." -> java`awt`event`KeyEvent`VKUPERIOD,

"'" -> java`awt`event`KeyEvent`VKUQUOTE,

";" -> java`awt`event`KeyEvent`VKUSEMICOLON,

"\[ShiftKey]" -> java`awt`event`KeyEvent`VKUSHIFT,

"/" -> java`awt`event`KeyEvent`VKUSLASH,

" " -> java`awt`event`KeyEvent`VKUSPACE,

"\[SpaceKey]" -> java`awt`event`KeyEvent`VKUSPACE,

"\[TabKey]" -> java`awt`event`KeyEvent`VKUTAB
}
