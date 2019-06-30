(* Wolfram Language package *)
BeginPackage["ZeroMQLink`HostParser`",{"ZeroMQLink`Libraries`"}]


addressBuildAssoc::usage="builds up an association of domain, port, etc."

addressAssocString::usage="takes an association from addressBuildAssoc and turns it into a canonical string form"


Begin["`Private`"]



(*this is the full function we use to parse the argument specifications for SocketOpen / SocketConnect and SocketListen (in some cases)*)
(*the first argument is the specification provided to the function, and the second is a default scheme to use - which is only really of interest for ZMQ sockets, tcp sockets use None for this*)
(*the third argument is the symbol to use for raising messages, the fourth is the protocol to assume, i.e. zmq or tcp, which really is just to identify the different minimum required information for the two different protocols*)
addressBuildAssoc[spec_,defaultScheme_,symbol_,protocol_]:=Block[{assoc,error},
	assoc = addressParse[spec];
	Which[assoc === $Failed,
		(*specification error*)
		(
			With[{s=symbol},Message[MessageName[s,"addrspec"],spec,"the domain or port is invalid"]];
			$Failed
		),
		assoc["Domain"] === $Failed,
		(*something specific about the domain is invalid*)
		(
			With[{s=symbol},Message[MessageName[s, "addrspec"],spec,"it has an invalid domain specification"]];
			$Failed
		),
		(*we're trying to do a normal tcp socket (evidenced by the None default scheme) and we didn't get a port, but we did get a valid scheme in the spec*)
		(*then we need to lookup the port number for the scheme using getservbyname*)
		protocol === "tcp" && defaultScheme === None && (assoc["Port"] === "*" || MissingQ[assoc["Port"]]) && KeyExistsQ["Scheme"]@assoc && assoc["Scheme"] =!= None, 
		(
			If[MatchQ[Missing["duplicate port",_?StringQ]]@assoc["Port"],
				(*THEN*)
				(*the port is duplicate due to conflicting scheme / port configuations, so issue a message and return*)
				(
					With[{s=symbol},Message[MessageName[s, "addrspec"],spec,"the port number conflicts with the scheme specification"]];
					$Failed
				),
				(*ELSE*)
				(*good to go to use the scheme for a lookup*)
				(
					assoc["Port"] = getservbyname[assoc["Scheme"],"tcp"];
					(*now if this is valid, we can continue as normal*)
					If[assoc["Port"] === $Failed,
						(*THEN*)
						(*invalid scheme service*)
						(
							With[{s=symbol},Message[MessageName[s, "addrspec"],spec,"it has an invalid service specification"]];
							$Failed
						),
						(*ELSE*)
						(*valid service from the scheme - can return what we got*)
						(
							assoc
						)
					]
				)
			]
		),
		(*on SocketConnect we can't use the port spec "*", so check if we're running for SocketConnect*)
		symbol === System`SocketConnect && assoc["Port"] === "*",
		(
			Message[System`SocketConnect::addrspec,spec,"the port must be a valid Integer for SocketConnect"];
			$Failed
		),
		MissingQ[assoc["Port"]],
		(*missing the port information or otherwise have some port problems*)
		(
			Switch[First[assoc["Port"]],
				"missing port",
				(
					With[{s=symbol},Message[MessageName[s, "addrspec"],spec,"it is missing the port specification"]];
					$Failed
				),
				"duplicate port",
				(
					(*note that for zmq sockets, if a user specifies an address like tcp://127.0.0.1:7000, that is valid, even though getservbyname will fail for tcp, leading to*)
					(*a "duplicate" port problem, however in this instance we can ignore this and use the second element of the Missing spec to use as the port*)
					If[protocol === "zmq" && Length[assoc["Port"]] === 2,
						(*THEN*)
						(*we're using a zmq socket, and so we should check to see if there is a second element to the spec, which means that we can proceed and just that getservbyname*)
						(*failed for the tcp transport layer specification*)
						(
							Append[assoc,"Port"->assoc["Port"][[2]]]
						),
						(*ELSE*)
						(*not zmq protocol, so fail*)
						(
							With[{s=symbol},Message[MessageName[s, "addrspec"],spec,"it has duplicate port specifications"]];
							$Failed
						)
					]
				),
				"invalid service",
				(
					With[{s=symbol},Message[MessageName[s, "addrspec"],spec,"it has an invalid service specification"]];
					$Failed
				)
			]
		),
		(*the scheme and port both exist aren't none*)
		KeyExistsQ["Scheme"]@assoc && assoc["Scheme"] =!= None,
		(
			assoc
		),
		(*if we get here we are handling zmq sockets, which need the scheme to not be null, as that corresponds to the transport layer*)
		defaultScheme =!= None,
		(
			Append[assoc,"Scheme"->defaultScheme]
		),
		protocol === "tcp" && !MissingQ[assoc["Port"]],
		(
			(*we don't need to specify a scheme in the string spec*)
			assoc
		),
		(*final fall through case to handle case where defaultScheme is None, i.e. tcp sockets, and we didn't get a port or scheme number from the parsing*)
		True,
		(
			With[{s=symbol},Message[MessageName[s, "addrspec"],spec,"it is missing port or service specification"]];
			$Failed
		)
	]
]


(*this helper function takes the association returned from addressBuildAssoc and turns it into a string spec, i.e. for zmq the full string is necessary, but for normal sockets, we just want the spec as the domain:port*)
addressAssocString[assoc_]:=If[KeyExistsQ["Scheme"]@assoc,
	StringDrop[URLBuild[KeyDrop["Port"]@assoc],-1]<>":"<>ToString[assoc["Port"]],
	assoc["Domain"]<>":"<>ToString[assoc["Port"]]
]


(*by default we use any available port on localhost - this form is only valid for SocketOpen, as you can't connect to any port on a different machine*)
addressParse[Automatic]:=<|"Domain"->"127.0.0.1","Port"->"*"|>;

(*this form is also only for SocketOpen, and is used for just specifying the port number*)
addressParse[port_?IntegerQ]:=<|"Scheme"->"tcp","Domain"->"127.0.0.1","Port"->ToString[port]|>;

(*this form expects spec to include the port, so if it doesn't we fail*)
addressParse[spec_]:=Block[
	{
		assoc=hostParse[spec],
		servicePort
	},
	If[assoc === $Failed || !AssociationQ[assoc],
		(*THEN*)
		(*failed to parse the host at all, so fail*)
		$Failed,
		(*ELSE*)
		(*was able to parse something*)
		If[KeyExistsQ["Port"]@assoc,
			(*THEN*)
			(*there was a port specified in the host spec, so now we just need to confirm that there wasn't also a conflicting scheme specified*)
			(
				If[KeyExistsQ["Scheme"]@assoc && assoc["Scheme"] =!= None,
					(*THEN*)
					(*we need to check the port number for the scheme, assigning missing if the service/scheme has a different port spec*)
					(
						servicePort = getservbyname[assoc["Scheme"],"tcp"];
						If[servicePort =!= readNum[assoc["Port"]],
							(*THEN*)
							(*the port and scheme conflicts, so mark the port as duplicate cause we don't know which one to use*)
							(
								Append[assoc,"Port"->Missing["duplicate port",assoc["Port"]]]
							),
							(*ELSE*)
							(*it agrees, we're good to just return it as is*)
							(
								assoc
							)
						]
					),
					(*ELSE*)
					(*no conflicting scheme spec, we can just return the assoc as found*)
					(
						assoc
					)
				]
			),
			(*ELSE*)
			(*there wasn't a port specified, so fail cause we don't know what port one to use*)
			Append[assoc,"Port"->Missing["missing port"]]
		]
	]
];

(*this form expects host to not include a port*)
addressParse[{host_,Automatic}]:=With[
	{assoc=hostParse[host]},
	If[assoc === $Failed || !AssociationQ[assoc],
		(*THEN*)
		(*failed to parse the host at all, so fail*)
		$Failed,
		(*ELSE*)
		(*was able to parse something*)
		If[!KeyExistsQ["Port"]@assoc,
			(*THEN*)
			(*there wasn't a port specified in the host spec, so we're good*)
			Append[assoc,"Port"->"*"],
			(*ELSE*)
			(*there was a port specified, so fail cause we don't know which one to use*)
			Append[assoc,"Port"->Missing["duplicate port"]]
		]
	]
];

(*this form expects host to not include a port as well*)
addressParse[{host_,port_?IntegerQ}]:=addressParse[{host,ToString[port]}];

(*this form uses an automatic port number*)
addressParse[{host_,"*"}]:=addressParse[{host,Automatic}]

(*this form can either have port be a string of a number,*)
(* or it can be a service such as "www" or "http", which we can't know deterministically from WL, so we have to drop down to C code and call getservbyname to get the appropriate port number*)
(*that's what serviceParse does for us*)
addressParse[{host_,str_?StringQ}]:=Block[
	{
		assoc = hostParse[host],
		num = readNum[str],
		servRes,
		servicePort
	},
	If[assoc === $Failed || !AssociationQ[assoc],
		(*THEN*)
		(*failed to parse the host at all, so fail*)
		$Failed,
		(*ELSE*)
		(*was able to parse something*)
		If[!KeyExistsQ["Port"]@assoc,
			(*THEN*)
			(*there wasn't a port specified in the host spec, so we're good to use the string*)
			If[NumberQ[num],
				(*THEN*)
				(*the string spec is a number and we can just use that as a number for comparison*)
				(*need to now check any scheme that may have been specified with the host spec*)
				If[KeyExistsQ["Scheme"]@assoc && assoc["Scheme"] =!= None,
					(*THEN*)
					(*the scheme exists and we need to check it*)
					(
						servicePort = getservbyname[assoc["Scheme"],"tcp"];
						If[servicePort =!= num,
							(*THEN*)
							(*the port and scheme conflicts, so mark the port as duplicate cause we don't know which one to use*)
							(
								Append[assoc,"Port"->Missing["duplicate port",""]]
							),
							(*ELSE*)
							(*it agrees, we're good to just return it as is*)
							(
								assoc
							)
						]
					),
					(*ELSE*)
					(*the scheme is missing, so we can just use the port number*)
					(
						Append[assoc,"Port"->ToString[num]]
					)
				],
				(*ELSE*)
				(*the string spec isn't a number, so assume it's a service specification, so need to use getservbyname*)
				(*note that the second arg to getservbyname is the protocol to use - we only support raw TCP sockets and ZMQ, which also uses tcp (if the transport protocol wasn't already specified as the scheme)*)
				(
					(*see if there was a scheme specified in the host, if there is we need to make sure they're the same*)
					If[KeyExistsQ["Scheme"]@assoc && assoc["Scheme"] =!= None,
						(*THEN*)
						(*scheme exists so we need to check it*)
						If[SameQ@@(servicePort=ToUpperCase@ToString@getservbyname[#,"tcp"]&/@{str,assoc["Scheme"]}),
							(*THEN*)
							(*the services both resolve to the same port number, so we're good*)
							(
								Append[assoc,"Port"->First[servicePort]]
							),
							(*ELSE*)
							(*the port specs from the services conflict*)
							Append[assoc,"Port"->Missing["duplicate port",str]]
						],
						(*ELSE*)
						(*scheme doesn't exist, so just continue as normal*)
						(
							servicePort = getservbyname[str,"tcp"];
							If[servicePort =!= $Failed,
								(*THEN*)
								(*the service port is good to use*)
								(
									Append[assoc,"Port"->servicePort]
								),
								(*ELSE*)
								(*the service lookup failed*)
								(
									Append[assoc,"Port"->Missing["invalid service"]]
								)
							]
						)
					]
				)
			],
			(*ELSE*)
			(*there was a port specified, so fail cause we dont know which one to use*)
			If[NumberQ[num],
				(*THEN*)
				(*the string spec is a number and we can just use that as a number for comparison*)
				(*need to now check any scheme that may have been specified with the host spec*)
				If[KeyExistsQ["Scheme"]@assoc && assoc["Scheme"] =!= None,
					(*THEN*)
					(*the scheme exists and we need to check it*)
					(
						servicePort = getservbyname[assoc["Scheme"],"tcp"];
						If[servicePort =!= num,
							(*THEN*)
							(*the service port and scheme conflicts, so mark the port as duplicate cause we don't know which one to use*)
							(
								Append[assoc,"Port"->Missing["duplicate port",""]]
							),
							(*ELSE*)
							(*it agrees, so finally we need to check to see if it agrees with the port specification from the host spec*)
							(
								If[ToString[servicePort] === ToString[assoc["Port"]] === ToString[num],
									(*THEN*)
									(*everything agrees*)
									assoc,
									(*ELSE*)
									(*there are differences, so fail*)
									Append[assoc,"Port"->Missing["duplicate port",""]]
								]
							)
						]
					),
					(*ELSE*)
					(*the scheme is missing, so just confirm that the port in the host agrees with the service port*)
					(
						If[ToString[num] === ToString[assoc["Port"]],
							(*THEN*)
							(*the service port exustsm confirm that it agrees with the *)
							(
								Append[assoc,"Port"->num]
							),
							(*ELSE*)
							(*the two disagree*)
							(
								Append[assoc,"Port"->Missing["duplicate port",""]]
							)
						]
					)
				],
				(*ELSE*)
				(*the string spec isn't a number, so assume it's a service specification, so need to use getservbyname*)
				(*note that the second arg to getservbyname is the protocol to use - we only support raw TCP sockets and ZMQ, which also uses tcp (if the transport protocol wasn't already specified as the scheme)*)
				(
					(*see if there was a scheme specified in the host, if there is we need to make sure they're the same*)
					If[KeyExistsQ["Scheme"]@assoc && assoc["Scheme"] =!= None,
						(*THEN*)
						(*scheme exists so we need to check it*)
						If[SameQ@@(servicePort=ToUpperCase@ToString@getservbyname[#,"tcp"]&/@{str,assoc["Scheme"]}),
							(*THEN*)
							(*the services both resolve to the same port number, so we're good*)
							(
								Append[assoc,"Port"->First[servicePort]]
							),
							(*ELSE*)
							(*the port specs from the services conflict*)
							Append[assoc,"Port"->Missing["duplicate port",str]]
						],
						(*ELSE*)
						(*scheme doesn't exist, so just continue as normal*)
						(
							servicePort = getservbyname[str,"tcp"];
							If[servicePort =!= $Failed,
								(*THEN*)
								(*the service port is good to use*)
								(
									Append[assoc,"Port"->servicePort]
								),
								(*ELSE*)
								(*the service lookup failed*)
								(
									Append[assoc,"Port"->Missing["invalid service"]]
								)
							]
						)
					]
				)
			]
		]
	]
];

addressParse[any___]:= $Failed

(*if there's ever an IPAddress or URL wrapper, just peel it off and recurse into it*)
hostParse[IPAddress[host_]]:=hostParse[host];
hostParse[URL[host_]]:=hostParse[host];

(*any other case we can't handle so we don't parse anything*)
hostParse[any___]:= $Failed

(*this is the main parsing function which first attempts to parse out the port, and if it finds one, it will remove it from the string*)
(*then it will attempt to parse out the scheme, removing the scheme from the url if it is present*)
(*ideally this would just call URLParse, but unfortunately it doesn't handle any kind of case that's not a "well formed URL", but we want to support people specifying not well formed URLs, so we have to do the parsing ourselves here*)
hostParse[host_?StringQ] :=
	(*first check if the host spec is a ipv4 or ipv6 address*)
	(*these functions are currently implemented in the kernel, they will eventually be moved into this package*) 
	If[Socket`IPv4AddressQ[host] || Socket`IPv6AddressQ[host],
		(*THEN*)
		(*we don't have any other specifications, so just return that host directly*)
		<|"Domain"->host|>,
		(*ELSE*)
		(*not a raw ipv4 or ipv6 address, so need to do the parsing manually*)
		With[{portRes = portParse[host]},
			If[portRes =!= $Failed,
				MapAt[checkDomain,Select[# =!= None &]@Join[portRes, schemeParse[portRes["Domain"]]],"Domain"],
				$Failed
			]
		]
	]
	

(*function to confirm the domain that's returned from the portParse/schemeParse functions is valid*)
checkDomain[domain_?StringQ]:=If[StringMatchQ[domain,StartOfString ~~ (WordCharacter|(WordCharacter .. ~~ ("." | "-")) ).. ~~ WordCharacter .. ~~ EndOfString],
	(*THEN*)
	(*valid domain, return it as is*)
	(
		domain
	),
	(*ELSE*)
	(*the domain is wrong - return $Failed*)
	(
		$Failed
	)
]

checkDomain[any___]:=$Failed

(*this will look for the first instance of :XXXX where XXXX is a sequence of entirely numbers, and assumes that this is the port*)
(*if it finds this, then it will return this as an Integer to be the port*)
(*NOTE : there is currently a bug/feature with this parsing, specifically for address's formed like "http:*:80", which currently*)
(*will parse as <|"Port" -> "*", "Domain" -> "http:80"|> which is wrong*)
(*this is because the first instance of ":"~~patt matches "http:*" *)
(*not sure if this can be solved. One way to fix it is to require addresses to include the // for schemes*)
portParse[spec_]:=
	With[
		{ports=StringCases[spec,":"~~port:(DigitCharacter..|"*")~~EndOfString:>port]},
		If[Length[ports]==1,
			(*THEN*)
			(*found at a port to use*)
			<|
				(*delete the : at the start of the port specifier, and then turn it into an integer*)
				(*note we only do this if it's not the any port specified "*" *)
				"Port"->If[First[ports]==="*","*",ToString[readNum[First[ports]]]],
				(*delete the port from the port, noting that there might be multiple instances of the same port string in the spec,*)
				(*so only delete the first one*)
				"Domain"->StringReplace[spec,":"<>First[ports]->"",1]
			|>,
			(*ELSE*)
			(*didn't find the port, see if there's something after the port spec, if there is fail cause we don't support that*)
			(
				If[Length[StringCases[spec,":"~~port:(DigitCharacter..|"*"):>port]] === 0,
					(*THEN*)
					(*no extra ones, so it just doesn't exist, which in some cases is okay and we can keep parsing*)
					<|"Port"->None,"Domain"->spec|>,
					(*ELSE*)
					(*found at least one but there's extra stuff in between the port and the end of the string, so fail*)
					$Failed
				]
			)
		]
	]

(*this will look for a sequence of letters at the start of the specification leading up to a :, and assumes that this is the scheme*)
(*note that this will most likely fail if there's a port number in the specification, as something like localhost:80 will match localhost as the port*)
schemeParse[spec_]:=
	With[
		(*note that although schemes is a list, there can only ever be 1, as we are using StartOfString*)
		(*also note that for cases where the full address spec is "localhost:", no port will have been parsed, and as such we receive here the domain of "localhost:", and need to *)
		(*not parse that as a scheme of localhost, we do that by making sure that there is something after the colon*)
		{schemes=StringCases[spec,scheme:(StartOfString~~LetterCharacter..)~~":"~~__~~EndOfString:>scheme]},
		If[Length[schemes]>=1,
			(*THEN*)
			(*successfully found a scheme, so return that*)
			<|
				"Scheme"->First[schemes],
				(*delete the scheme from the string and use that as the domain, and if there was a // after the scheme like in tcp://localhost, that will still be left in the string, so delete that too*)
				"Domain"->Fold[StringReplace[#1,#2,1]&,spec,{First[schemes]->"",StartOfString ~~ ("://" | ":"):>""}]
			|>,
			(*ELSE*)
			(*didn't find a scheme, just return the Domain as the spec*)
			<|"Scheme"->None,"Domain"->spec|>
		]
	]
	
	
(*helper function that lets us safely and quickly evaluate a string representation of a number into a string - note this works for both integers and real numbers, but not complex numbers*)
readNum[str_] := Block[{strm = StringToStream[str], res}, res = Quiet[Read[strm,Number]/.EndOfFile->$Failed,Read::readn]; Close[strm]; res]

getservbyname[service_?StringQ,protocol_?StringQ]:=Block[{res},
	(
		res = igetservbyname[service,protocol];
		If[First[res] =!= 0,
			$Failed,
			Last[res]
		]
	)
]



End[]

EndPackage[]
