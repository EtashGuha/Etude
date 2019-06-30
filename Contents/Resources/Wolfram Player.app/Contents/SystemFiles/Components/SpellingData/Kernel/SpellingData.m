BeginPackage["SpellingData`"]

SpellingDataDicts[] := Association[
	"English" -> "en_US",
	"French" -> "fr",
	"German" -> "de_DE",
	"Italian" -> "it_IT", (* ??? *)
	"Polish" -> "pl_PL",
	"Russian" -> "ru_RU",
	"Spanish" -> "es_ANY",
	"Ukrainian" -> "uk_UA",
	"Korean" -> "ko-KR",
	"Afrikaans" -> "af_ZA",
	"Aragonese" -> "an_ES",
	"Arabic" -> "ar",
	"Belarusan" -> "be_BY",
	"Breton" -> "br_FR",
	"Bosnian" -> "bs_BA",
	"Catalan" -> "ca",
	"Danish" -> "da_DK",
	"Greek" -> "el_GR",
	"Estonian" -> "et_EE",
	"Hebrew" -> "he_IL",
	"Croatian" -> "hr_HR",
	"Hungarian" -> "hu_HU",
	"Icelandic" -> "is",
	"Kurdish" -> "kmr_Latn",
	"Lao" -> "lo_LA",
	"Lithuanian" -> "lt"(*"lt_LT"?*),
	"Latvian" -> "lv_LV",
	"Nepali" -> "ne_NP",
	"Dutch" -> "nl_NL",
(*	"BrazilianPortuguese" -> "pt_BR",*)
	"Romanian" -> "ro_RO"(*"ro"*),
	"Slovak" -> "sk_SK",
	"Slovenian" -> "sl_SI",
	"Serbian" -> "sr",
	"Swedish" -> "sv_SE",
	"Swahili" -> "sw_TZ",
	"Telugu" -> "te_IN",
	"Thai" -> "th_TH"
]

SetAttributes[{SpellingDataDicts}, {Protected, ReadProtected}]

EndPackage[];