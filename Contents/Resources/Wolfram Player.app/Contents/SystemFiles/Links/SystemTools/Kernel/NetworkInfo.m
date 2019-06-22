(* ::Package:: *)

Begin["SystemTools`"];

System`$NetworkConnected;
System`DomainRegistrationInformation;
System`DomainRegisteredQ;

SystemTools`$WhoisErrorTLDResponse;
SystemTools`$WhoisErrorRegistryResponse;
SystemTools`$WhoisErrorRegistrarResponse;

DomainRegistrationInformation::noparse = "Unable to parse domain registration information. Returning unparsed WHOIS responses.";
DomainRegistrationInformation::noregis = "The provided domain name is not registered.";
DomainRegistrationInformation::nofqdn = "The provided domain name (`1`) is incomplete.";
DomainRegistrationInformation::unrectld = "The top-level domain (`1`) of the provided domain name is not recognized.";

Unprotect[$NetworkConnected, DomainRegistrationInformation];

Begin["`Private`"];

keyRename[a_, old_ -> new_] := KeyDrop[old][Append[a, new -> a[old]]];
keysRename[a_, reps_] := Fold[keyRename, a, reps];

If[!StringQ[$libraryPath],
	$libraryPath = FindLibrary["libSystemTools"];
	If[$libraryPath === $Failed,
		Message[LibraryFunction::load, "libSystemTools"];
		,
		If[LibraryLoad[$libraryPath] === $Failed,
			Message[LibraryFunction::load, $libraryPath];
		]
	]
]

(* Network utilities *)

networkQ = LibraryFunctionLoad[$libraryPath, "networkQ", {}, {"Boolean"}];

$NetworkConnected := networkQ[];

(* Whois Utility *)

(* Helper functions *)

whoisDomain[server_String, domain_String, return_String] := Module[{wh, rs},
	wh = Quiet[SocketConnect[server <> ":43", "TCP"]];
	If[wh === $Failed, Return[""]];

	Quiet[WriteString[wh, domain <> return]];

	rs = Quiet[TimeConstrained[ReadString[wh, TimeConstraint -> 2], 3]];

	If[rs === $Aborted || rs === $Failed,
		If[return != "\r\n",
			rs = whoisDomain[server, domain, "\r\n"];,
			rs = "";
		];
	];

	Quiet[Close[wh]];
	Return[rs];
];

whoisDomain[server_String, domain_String] := whoisDomain[server, domain, "\n"];

orderAssoc[assoc_Association, order_List] :=
	Module[
		{keyF, match2Plac, keyMatchListQ, keyH},

		match2Plac = AssociationThread[order -> Range[Length[order]]];
		AssociateTo[match2Plac, -1 -> 109];

		keyMatchListQ[mList_List, key_String] := AllTrue[mList, (StringContainsQ[key, #1] &)];
		keyF[key_String] := match2Plac[SelectFirst[order, (keyMatchListQ[#1, ToLowerCase[key]] &), -1]];
		keyF[key_] := 109;
		keyH[key_String] := (keyMatchListQ[#1, ToLowerCase[key]] &) /@ order;

		(*

		Print["TRUTH VALUES"];
		Print[keyH/@Keys[assoc]];
		Print["PLACEMENT VALUES"];
		Print[AssociationThread[Keys[assoc]\[Rule] keyF/@Keys[assoc]]];

		*)

		KeySortBy[assoc, keyF]
	];

parseDate[rDate_String] :=
	Module[{res, date},
		date = StringJoin[Riffle[Drop[StringSplit[rDate, "-"], -1], "-"]];
		res = Quiet[DateObject[date]];
		If[Unevaluated[DateObject[date]] === res,
			date,
			res
		]
	];

parseDate[rDate_] := rDate;

(* RDAP is the successor to WHOIS. It provides data in a more machine-friendly format. *)
(* However, for the average person, in most cases it provides zero, say, ownership information (something WHOIS itself provides). *)
(* This may change in the future as it becomes more developed. *)
(* Currently, the DomainRegistrationInformation function utilizes both RDAP and actual WHOIS. *)
getRDAPInfo[domain_String] :=
Module[
	{RDAPInfo, RDAPAssoc, nsInfoList, enLangs},

	RDAPInfo = Quiet[Import["https://rdap-pilot.verisignlabs.com/rdap/v1/domain/" <> domain, "JSON"]];
	If[FailureQ[RDAPInfo],
		Return[Association[]];
	];

	RDAPInfo = 
		Association[
			FixedPoint[
				Replace[
					#1, 
					x_List /; AllTrue[x, (Head[#1] === Rule) &] -> Association[x],
					Infinity
				] &, RDAPInfo
			]
		];

	RDAPAssoc = Association[];

	(*
	If[KeyExistsQ[RDAPInfo, "ldhName"],
	AssociateTo[RDAPAssoc, "Object Name" -> RDAPInfo["ldhName"]]
	];
	If[KeyExistsQ[RDAPInfo, "objectClassName"],
	AssociateTo[RDAPAssoc, 
	"Object Class" -> RDAPInfo["objectClassName"]]
	];
	*)

	If[KeyExistsQ[RDAPInfo, "lang"],
		If[!MissingQ[(enLangs = EntityList[EntityClass["LanguageLocale", "IETFTag" -> RDAPInfo["lang"]]])] && Length[enLangs] > 0,
			AssociateTo[RDAPAssoc, "Language" -> First[enLangs]];
			,
			AssociateTo[RDAPAssoc, "Language" -> RDAPInfo["lang"]];
		];
	];

	If[KeyExistsQ[RDAPInfo, "port43"],
		AssociateTo[RDAPAssoc, "RDAPServer" -> RDAPInfo["port43"]]
	];

	If[KeyExistsQ[RDAPInfo, "nameservers"],
		nsInfoList =
			Association[
				(*
				If[KeyExistsQ[#1, "ldhName"],
				"Object Name" -> #1["ldhName"],
				Nothing
				],
				If[KeyExistsQ[#1, "objectClassName"],
				"Object Class" -> #1["objectClassName"],
				Nothing
				],
				*)
				If[KeyExistsQ[#1, "ipAddresses"],
					"IPAddress" -> Replace[#1["ipAddresses"], (li_List/;Length[li] > 0) :> IPAddress[First[li]], Infinity],
					Nothing
				],
				If[KeyExistsQ[#1, "status"],
					"Status" -> #1["status"][[1]],
					Nothing
				]
			] & /@ RDAPInfo["nameservers"];
		AssociateTo[RDAPAssoc, "Nameservers" -> nsInfoList];
	];
	Return[RDAPAssoc];
];

(* Trim and order category keys *)
trimAssocCat[assoc_Association, trimStr_String] := Module[{catKeys, catKeysReformed, catKeysReps},
	catKeys = Keys[assoc];
	catKeysReformed = Replace[StringTrim[#1, trimStr]& /@ catKeys, "" -> "Name", {1}];
	catKeysReps = MapThread[Rule, {catKeys, catKeysReformed}];
	Return[
		Dataset[
			orderAssoc[
				keysRename[assoc, catKeysReps],
				{{"name"}, {"organisation"}, {"organization"}, {"url"}, {"whois"}}
			]
		]
	];
];

(* Helper functino to getPureWhoisInfo *)
(* Classify pure whois assoc by registry, registrar, registrant, and events *)
classifyPureWhois[pureWhoisAssoc_Association] :=
	Module[
		{
			registryRule,
			registrarRule,
			registrantRule,
			eventRule
		},

		registryRule =
			"Registry" -> 
				KeySelect[
					pureWhoisAssoc,
					StringQ[#1] && StringMatchQ[#1, "Registry" ~~ ___] &
				];

		registryRule[[2]] = trimAssocCat[registryRule[[2]], "Registry "];

		If[Length[registryRule[[2]]] < 1,
			registryRule = Nothing;
		];

		registrarRule = 
			"Registrar" -> 
				KeySelect[
					pureWhoisAssoc,
					StringQ[#1] && StringMatchQ[#1, "Registrar" ~~ ___] &
				];

		registrarRule[[2]] = trimAssocCat[registrarRule[[2]], "Registrar "];

		If[Length[registrarRule[[2]]] < 1,
			registrarRule = Nothing;
		];

		registrantRule = 
			"Registrant" -> 
				KeySelect[
					pureWhoisAssoc,
					StringQ[#1] && StringMatchQ[#1, "Registrant" ~~ ___] &
				];

		registrantRule[[2]] = trimAssocCat[registrantRule[[2]], "Registrant "];

		If[Length[registrantRule[[2]]] < 1,
			registrantRule = Nothing;
		];

		If[
			KeyExistsQ[pureWhoisAssoc, "Creation Date"] &&
				KeyExistsQ[pureWhoisAssoc, "Updated Date"],
			eventRule =
				"Events" -> Dataset[
					{
						Association[
							"Creation Date" -> pureWhoisAssoc["Creation Date"], 
							"Updated Date" -> pureWhoisAssoc["Updated Date"]
						]
					}
				];
			,
			eventRule = Nothing;
		];

		Return[Association[eventRule, registryRule, registrarRule, registrantRule]];
	];

(* getNonUSWhoisInfo[domain_String] := Module[{tldWhoisResponse, tldWhoisAssoc,reformedTLDAssoc},
  tldWhoisResponse = whoisDomain["whois.iana.org", domain];
  tldWhoisAssoc = StringCases[tldWhoisResponse,
 StartOfLine ~~ Shortest[regtype___] ~~ ":" ~~ Shortest[___] ~~ 
   Whitespace ~~ StartOfLine ~~ Whitespace ~~ Shortest[key___] ~~ 
   ":" ~~ Whitespace ~~ Shortest[value___] ~~ EndOfLine :> 
  Rule[StringJoin[StringTrim[regtype, Whitespace], " ", 
    StringTrim[key, Whitespace]], StringTrim[value, Whitespace]], 
 Overlaps -> True];

    Return[tldWhoisAssoc];
] *)

intPhoneNum[phone_]:=Module[{revPhone},
		(* Check time constraint here: is 5 seconds reasonable? *)
		revPhone = Quiet[TimeConstrained[Interpreter["PhoneNumber"][phone], 5]];
		If[FailureQ[revPhone],
			Return[phone];
		];
		Return[revPhone];
	];

intWrapper[class_, input_]:=Module[{revInput},
		(* Check time constraint here: is 5 seconds reasonable? *)
		revInput = Quiet[TimeConstrained[Interpreter[class][input], 5]];
		If[FailureQ[revInput],
			Return[input];
		];
		Return[revInput];
	];

(* getPureWhoisInfo queries WHOIS servers concerning a domain, and follows their referrals *)
getPureWhoisInfo[domain_String] :=
	Module[
		{
			containsN,
			WNull,

			reformedTLDAssoc,

			tldWhoisResponse,
			tldWhoisAssoc,

			registryWhoisServer,
			registryWhoisResponse,
			registryWhoisAssoc,

			registrarWhoisServer,
			registrarWhoisResponse,
			registrarWhoisAssoc,

			mrgRes,
			mKeys,

			mrgResKeys, expiryKeys, expiryKeysReformed, expiryKeysReps,

			urlKeys, emailKeys, phoneKeys, phoneKey, faxKeys, faxKey
		},

		containsN = ! StringContainsQ[#1, "\n"] &;
		WNull = Longest[Whitespace?containsN | ""];

		tldWhoisResponse = whoisDomain["whois.iana.org", domain];
		SystemTools`$WhoisErrorTLDResponse = tldWhoisResponse;

		tldWhoisAssoc = Association[
			StringCases[
				tldWhoisResponse,
				StartOfLine ~~ Whitespace ~~ Shortest[key___] ~~ ":" ~~ WNull ~~ Shortest[value___] ~~ WNull ~~ EndOfLine :> 
					(key -> StringTrim[value, "\r"])
			]
		];

		registryWhoisServer = tldWhoisAssoc["refer"];

		If[MissingQ[registryWhoisServer],
			reformedTLDAssoc = Association[];

			If[KeyExistsQ[tldWhoisAssoc, "domain"],
				AssociateTo[
					reformedTLDAssoc,
					{
						"Object Name" -> tldWhoisAssoc["domain"],
						"Object Class" -> "Domain"
					}
				];
			];

			If[KeyExistsQ[tldWhoisAssoc, "organisation"],
				AssociateTo[
					reformedTLDAssoc,
					"Registry Organization" -> tldWhoisAssoc["organisation"]
				];
			];

			If[KeyExistsQ[tldWhoisAssoc, "whois"],
				AssociateTo[
					reformedTLDAssoc,
					"Registry WHOIS Server" -> tldWhoisAssoc["whois"]
				];
			];

			Return[classifyPureWhois[reformedTLDAssoc]];
		];

		If[KeyExistsQ[tldWhoisAssoc, "organisation"],
			AssociateTo[
				tldWhoisAssoc,
				"Registry Organization" -> tldWhoisAssoc["organisation"]
			];
		];

		tldWhoisAssoc = 
			Merge[
				{tldWhoisAssoc, Association["Registry WHOIS Server" -> tldWhoisAssoc["refer"]]},
				#1[[1]] &
			];

		registryWhoisResponse = whoisDomain[registryWhoisServer, domain];
		SystemTools`$WhoisErrorRegistryResponse = registryWhoisResponse;

		If[StringContainsQ[registryWhoisResponse, "NO MATCH" | "NOT FOUND", IgnoreCase -> True],
			Message[DomainRegistrationInformation::noregis];
			If[TrueQ[regisCheck], isRegis = False;];
			Abort[];
		];

		If[TrueQ[regisCheck], 
			If[StringMatchQ[tldWhoisAssoc["domain"], "COM" | "NET" | "ORG", IgnoreCase -> True],
				isRegis = True;
			];
			Abort[];
		];

		registryWhoisAssoc = Association[
			StringCases[
				registryWhoisResponse,
				StartOfLine ~~ WNull ~~ Shortest[key___] ~~ ":" ~~ WNull ~~ Shortest[value___] ~~ WNull ~~ EndOfLine :> 
					(key -> StringTrim[value, "\r"])
			]
		];

		registrarWhoisServer = registryWhoisAssoc["Registrar WHOIS Server"];

		If[MissingQ[registrarWhoisServer],
			Return[
				classifyPureWhois[Merge[{registryWhoisAssoc, tldWhoisAssoc}, #1[[1]] &]]
			];
		];

		registrarWhoisResponse = whoisDomain[registrarWhoisServer, domain];
		SystemTools`$WhoisErrorRegistrarResponse = registrarWhoisResponse;

		registrarWhoisAssoc = 
			Association[
			StringCases[
				registrarWhoisResponse,
				StartOfLine ~~ WNull ~~ (Shortest[key___]/;!StringContainsQ[key, "Ext"]) ~~ ":" ~~ WNull ~~ Shortest[value___] ~~ WNull ~~ EndOfLine :> 
					(StringReplace[key, {"Admin" -> "Registrant Admin", "Tech" -> "Registrant Tech"}] -> StringTrim[value, "\r"])
			]
		];

		mrgRes =  Merge[{registrarWhoisAssoc, registryWhoisAssoc, tldWhoisAssoc}, #1[[1]] &];

		mrgResKeys = Keys[mrgRes];
		expiryKeys = Select[mrgResKeys, StringContainsQ[#1, "Expiry"]&];
		expiryKeysReformed = StringReplace[#1, "Expiry" -> "Expiration"]& /@ expiryKeys;
		expiryKeysReps = MapThread[Rule, {expiryKeys, expiryKeysReformed}];

		mrgRes = keysRename[mrgRes, expiryKeysReps];

		tooLongKeys = Keys[Select[mrgRes, (StringLength[#1] > 95 || StringLength[#1] == 0)&]];
		KeyDropFrom[mrgRes, tooLongKeys];

		mKeys = Keys[KeySelect[mrgRes, StringContainsQ[#1, "Date", IgnoreCase -> True] &]];
		AssociateTo[
			mrgRes,
			(#1 -> parseDate[mrgRes[#1]])& /@ mKeys
		];

		urlKeys = Keys[KeySelect[mrgRes, StringContainsQ[#1, "URL", IgnoreCase -> True] &]];
		AssociateTo[
			mrgRes,
			(#1 -> URL[mrgRes[#1]])& /@ urlKeys
		];

		emailKeys = Keys[KeySelect[mrgRes, StringContainsQ[#1, "Email", IgnoreCase -> True] &]];
		AssociateTo[
			mrgRes,
			(#1 -> Hyperlink[mrgRes[#1], "mailto:" <> mrgRes[#1]])& /@ emailKeys
		];

		phoneKeys = Keys[KeySelect[mrgRes, StringContainsQ[#1, "Phone", IgnoreCase -> True] &]];
		AssociateTo[
			mrgRes,
			(phoneKey = intPhoneNum[mrgRes[#1]]; #1 -> Hyperlink[phoneKey, "tel:" <> phoneKey])& /@ phoneKeys
		];

		faxKeys = Keys[KeySelect[mrgRes, StringContainsQ[#1, "Fax", IgnoreCase -> True] &]];
		AssociateTo[
			mrgRes,
			(faxKey = intPhoneNum[mrgRes[#1]]; #1 -> Hyperlink[faxKey, "fax:" <> faxKey])& /@ faxKeys
		];

		countryKeys = Keys[KeySelect[mrgRes, StringContainsQ[#1, "Country", IgnoreCase -> True] &]];
		AssociateTo[
			mrgRes,
			(#1 -> intWrapper["Country", mrgRes[#1]])& /@ countryKeys
		];

		stateProvinceKeys = Keys[KeySelect[mrgRes, StringContainsQ[#1, "State" | "Province", IgnoreCase -> True] &]];
		AssociateTo[
			mrgRes,
			(#1 -> intWrapper["AdministrativeDivision", mrgRes[#1]])& /@ stateProvinceKeys
		];

		Return[
			classifyPureWhois[
				mrgRes
			]
		];
	];

getRDAPIPInfo[ip_String] := Module[{fnCases, RDAPAssoc, nsInfoList, RDAPInfo, ownershipInfo, eventsInfo}, 
	RDAPInfo = Quiet[Import["https://rdap.arin.net/registry/ip/" <> ip, "JSON"]];
	If[FailureQ[RDAPInfo], Return[$Failed];];

	RDAPInfo = 
		Association[
			FixedPoint[
				Replace[
					#1, 
					x_List /; AllTrue[x, (Head[#1] === Rule) &] -> Association[x],
					Infinity
				]&,
				RDAPInfo
			]
		];

	RDAPAssoc = Association[];

	If[KeyExistsQ[RDAPInfo, "objectClassName"], 
		AssociateTo[RDAPAssoc, "Object Class" -> "IP Network"];
	];

	If[KeyExistsQ[RDAPInfo, "name"],
		AssociateTo[RDAPAssoc, "Corresponding Name" -> RDAPInfo["name"]]
	];

	If[KeyExistsQ[RDAPInfo, "port43"], 
		AssociateTo[RDAPAssoc, "RDAPServer" -> RDAPInfo["port43"]]
	];

	If[KeyExistsQ[RDAPInfo, "ipVersion"], 
		AssociateTo[RDAPAssoc, "IP Version" -> RDAPInfo["ipVersion"]]
	];

	If[KeyExistsQ[RDAPInfo, "startAddress"], 
		AssociateTo[RDAPAssoc, "Start of Address Range" -> RDAPInfo["startAddress"]]
	];

	If[KeyExistsQ[RDAPInfo, "endAddress"], 
		AssociateTo[RDAPAssoc, "End of Address End" -> RDAPInfo["endAddress"]]
	];

	If[
		KeyExistsQ[RDAPInfo, "events"],
		eventsInfo = 
			If[
				AssociationQ[#1] && KeyExistsQ[#1, "eventAction"] && KeyExistsQ[#1, "eventDate"],
				Association[
					"Event Action" -> Capitalize[#1["eventAction"], "AllWords"],
					"Event Date" -> parseDate[#1["eventDate"]]
				],
				Nothing
			] & /@ RDAPInfo["events"];
		If[Length[eventsInfo] > 0,
			AssociateTo[RDAPAssoc, "Events" -> eventsInfo];
		];
	];

	If[
		KeyExistsQ[RDAPInfo, "entities"] &&
			Length[RDAPInfo["entities"]] > 0 &&
			KeyExistsQ[RDAPInfo["entities"][[1]], "vcardArray"] &&
			Length[RDAPInfo["entities"][[1]]["vcardArray"]] > 1
		,
		ownershipInfo = Association[];
		fnCases = Cases[
			RDAPInfo["entities"][[1]]["vcardArray"][[2]],
			{"fn", _, _, owner_String, ___} -> owner
		];
		If[Length[fnCases] > 0,
			AssociateTo[ownershipInfo, "Owner Name" -> fnCases[[1]]];
		];

		adrCases = Cases[
			RDAPInfo["entities"][[1]]["vcardArray"][[2]],
			{"adr", ownerAdr_Association, ___} -> ownerAdr
		];
		If[Length[adrCases] > 0,
			If[KeyExistsQ[adrCases[[1]], "label"],
				AssociateTo[ownershipInfo, "Owner Address" -> adrCases[[1]]["label"]];
			];
		];

		If[Length[ownershipInfo] > 0,
			AssociateTo[RDAPAssoc, "Ownership Information" -> Dataset[ownershipInfo]];
		];
	];

	If[Length[RDAPAssoc] > 0,
		Return[RDAPAssoc];,
		Return[$Failed];
	];
];

pruneDomain[dom_String]:=
	Module[{suffs, pattSuffs, noSuffs, plainSuffs, car, cdr},
		suffs = Cases[
					ReadList["https://publicsuffix.org/list/public_suffix_list.dat","String"],
					suff_String /; !StringMatchQ[suff, "//" ~~ ___] :> suff
				];
		pattSuffs = Cases[suffs,suff_ /; StringContainsQ[suff,"*"] :> StringExpression@@StringSplit[suff,"*" ->  ___]];
		noSuffs = Cases[suffs,suff_ /; StringContainsQ[suff,"!"] :> First[StringCases[suff,"!" ~~ exp___ -> exp]]];
		plainSuffs = Cases[suffs,suff_ /; !StringContainsQ[suff,"*"|"!"]];

		If[StringMatchQ[dom, Alternatives @@ noSuffs],
			Return[dom];
		];

		If[
			StringMatchQ[dom,(Alternatives @@ Join[pattSuffs,plainSuffs]) ~~ EndOfLine],
			Message[DomainRegistrationInformation::nofqdn, dom];
			Return[$Failed];
		];

		If[!StringMatchQ[dom, ___ ~~ "." ~~(Alternatives @@ Join[pattSuffs,plainSuffs]) ~~ EndOfLine],
			Message[DomainRegistrationInformation::unrectld, Last[StringSplit[dom, "."]]];
			Return[$Failed];
		];

		car = First[StringSplit[dom,"." ~~ (Alternatives @@ Join[pattSuffs,plainSuffs]) ~~ EndOfLine]];
		cdr = StringTrim[dom, car];

		Return[StringJoin[Riffle[Reverse[Take[Reverse[StringSplit[car, "."]], UpTo[1]]], "."], cdr]];
	];

DomainRegisteredQ[input_String] := Block[{regisCheck = True, isRegis = $Failed}, Quiet[DomainRegistrationInformation[input]]; isRegis];

DomainRegistrationInformation[input_String] :=
	Module[
		{domain, whoisAssoc},

			SystemTools`$WhoisErrorTLDResponse = Null;
			SystemTools`$WhoisErrorRegistryResponse = Null;
			SystemTools`$WhoisErrorRegistrarResponse = Null;

			If[
				StringMatchQ[
					input, 
					NumberString ~~ "." ~~ NumberString ~~ "." ~~ NumberString ~~ "." ~~ NumberString ~~ ("." | "")
				],
				(* Return[Replace[getRDAPIPInfo[input], Association[x___] :>  Dataset[Association[x]], {0, Infinity}]]; *)
				Return[Dataset[getRDAPIPInfo[input]]];
				,
				domain = pruneDomain[input];

				If[FailureQ[domain],
					Return[$Failed];
				];

				whoisAssoc = CheckAbort[Merge[{getRDAPInfo[domain], getPureWhoisInfo[domain]}, #1[[1]] &], $Failed];
				If[FailureQ[whoisAssoc], Return[$Failed]];

				whoisAssoc = 
					Merge[
						{
							Association["DomainName" -> ToLowerCase[domain] (* "Object Name" -> ToUpperCase[domain], "Object Class" -> "Domain" *)],
							whoisAssoc
						},
						#1[[1]]&
					];

				If[
					Length[whoisAssoc] < 4,
					Message[DomainRegistrationInformation::noparse];
					Return[
						Association[
							"RegistryResolverResponse" -> SystemTools`$WhoisErrorTLDResponse, 
							"RegistryResponse" -> SystemTools`$WhoisErrorRegistryResponse,
							"RegistrarResponse" -> SystemTools`$WhoisErrorRegistrarResponse
						]
					];
				];
				(* Return[Replace[whoisAssoc, Association[x___] :>  Dataset[Association[x]], {0, Infinity}]]; *)

				Return[
					Dataset[orderAssoc[whoisAssoc, {{"domain", "name"},{"event"},{"registrant"},{"registrar"},{"registry"},{"name","server"}}]]
				];
			];
	];

(* START: Add to ServiceExecute *)

ServiceConnections`Private`externalservice["DomainRegistrationInformation","Name"]:="DomainRegistrationInformation";
ServiceConnections`Private`externalservice["DomainRegistrationInformation","Requests"]:={"Lookup"(*,"Whois Lookup", "RDAP Lookup"*)};
ServiceConnections`Private`externalservice["DomainRegistrationInformation","Information"]:="Lookups a domain's registration information.";
ServiceConnections`Private`externalservice["DomainRegistrationInformation","ID"]:=0;
ServiceConnections`Private`externalservice["DomainRegistrationInformation","Lookup",rl_Rule]:=ServiceConnections`Private`externalservice["DomainRegistrationInformation","Lookup",{rl}];
ServiceConnections`Private`externalservice["DomainRegistrationInformation","Lookup",{"Domain"->dmn_String}]:=DomainRegistrationInformation[dmn];

(* END: Add to ServiceExecute *)

Protect[$NetworkConnected, DomainRegistrationInformation];

End[];
End[];
