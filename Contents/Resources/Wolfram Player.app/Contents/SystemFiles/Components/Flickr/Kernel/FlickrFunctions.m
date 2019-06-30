BeginPackage["FlickrFunctions`"];

getLoginInfo::usage = "";
getUserId::usage = "";
$userCamelFields::usage = "";
getUserIconUrl::usage = "";
getphotourl::usage = "";
getPhotoMainURL::usage = "";
getPhoto::usage = "";
getOriginalPhoto::usage = "";
mapImageSizes::usage = "";
formatImageFullData::usage = "";
baseEncode::usage = "";
parseDates::usage = "";
formatDate::usage = "";
formatDate2::usage = "";
formatDate3::usage = "";
formatTime::usage = "";

Begin["`Private`"];

getLoginInfo[id_]:= With[{res = Quiet[Developer`ReadRawJSONString[OAuthClient`rawoauthdata[id,"RawTestLogin",{"format" -> "json", "nojsoncallback" -> "1"}]]]},
	If[AssociationQ[res] && (res["stat"] =!= "fail"),
		getLoginInfo[id] = res,
		Message[ServiceExecute::nval,"User","Flickr"];
		Throw[$Failed]
	]
]

getUserIconUrl[iconFarm_,iconServer_,userId_]:= "https://www.flickr.com/images/buddyicon.gif" /; (ToExpression[iconServer]>0)
getUserIconUrl[iconFarm_,iconServer_,userId_]:= "http://farm"<>ToString[iconFarm]<>".staticflickr.com/"<>ToString[iconServer]<>"/buddyicons/"<>ToString[userId]<>".jpg"

getUserId[userParam_,id_] := Block[{userinfo, userid = ""},
	Switch[userParam,
			_?StringQ,
				userinfo = ImportString[OAuthClient`rawoauthdata[id,"RawFindUserByUsername",{"username"->userParam,"format"->"json","nojsoncallback"->"1"}],"RawJSON"];
				If[userinfo["stat"] === "ok", userid = userinfo["user"]["nsid"]],
			{"UserID", _?StringQ},
				userid = userParam[[2]],
			{"UserName", _?StringQ},
				userinfo = ImportString[OAuthClient`rawoauthdata[id,"RawFindUserByUsername",{"username"->userParam[[2]],"format"->"json","nojsoncallback"->"1"}],"RawJSON"];
				If[userinfo["stat"] === "ok", userid = userinfo["user"]["nsid"]],
			{"UserEmail", _?StringQ},
				userinfo = ImportString[OAuthClient`rawoauthdata[id,"RawFindUserByEmail",{"find_email"->userParam[[2]],"format"->"json","nojsoncallback"->"1"}],"RawJSON"];
				If[userinfo["stat"] === "ok", userid = userinfo["user"]["nsid"]]
		];
	userid
]

$userCamelFields = Dispatch[{"can_buy_pro" -> "CanBuyPro", "contact" -> "IsContact", "count" -> "Count", "description" -> "Description", 
 "expire" -> "Expire", "family" -> "IsFamily", "firstdate" -> "FirstDate", "firstdatetaken" -> "FirstDateTaken", "friend" -> "IsFriend", "gender" -> "Gender", 
 "has_stats" -> "HasStats", "id" -> "ID", "ignored" -> "Ignored", "ispro" -> "IsPro", "label" -> "Label", "location" -> "Location", "mbox_sha1sum" -> "MboxSha1sum",
 "mobileurl" -> "MobileURL", "nsid" -> "NSID", "offset" -> "Offset", "path_alias" -> "PathAlias", "photos" -> "Photos", "photosurl" -> "PhotosURL", "profileurl" -> "ProfileURL",
 "realname" -> "RealName", "timezone" -> "Timezone", "timezone_id" -> "TimezoneID", "username" -> "Username", "views" -> "Views"}]

getphotourl[farmid_,serverid_,id_,secret_,size_:"t"] := "http://farm" <> ToString[farmid] <> ".staticflickr.com/" <> ToString[serverid] <> "/" <> 
  ToString[id] <> "_" <> ToString[secret] <> "_" <> ToString[size] <> ".jpg";

getPhotoMainURL[info_] := "https://www.flickr.com/photos/" <> ToString[info["owner"]] <> "/" <> 
  ToString[info["id"]];

getPhoto[info_, size_: "t", import_:False] := Module[{farm, server, id, secret, url},
	farm = ToString[info["farm"]];
	server = ToString[info["server"]];
	id = ToString[info["id"]];
	secret = ToString[info["secret"]];
	url = "http://farm" <> farm <> ".staticflickr.com/" <> server <> "/" <> id <> "_" <> secret <> "_" <> size <> ".jpg";
	If[import,Image[Import[url],MetaInformation->{"Source"->Hyperlink["https://flic.kr/p/"<>baseEncode[id]]}],url]
]

getOriginalPhoto[info_, import_:False] := Module[{farm, server, id, secret, url, format},
	farm = ToString[info["farm"]];
	server = ToString[info["server"]];
	id = ToString[info["id"]];
	secret = info["originalsecret"];
	If[MissingQ["original"],
		url = ServiceExecute["Flickr", "RawImageSizes", {"format" -> "json", "nojsoncallback" -> "1", "photo_id" -> id}]["sizes"]["size"][[-1, "source"]],
		format = ToString[Lookup[info,"original_format","jpg"]];
		url = "http://farm" <> farm <> ".staticflickr.com/" <> server <> "/" <> id <> "_" <> secret <> "_" <> "o" <> "." <> format
	];
	If[import,Image[Import[url],MetaInformation->{"Source"->Hyperlink["https://flic.kr/p/"<>baseEncode[id]]}],url]
]

mapImageSizes[size_] := size /. {"Small" -> "n", "Medium" -> "c", "Large" -> "b", "Thumbnail" -> "t", "Original" -> "o"};

formatImageFullData[data_] := Block[{result=data, lat, lon},
   If[KeyExistsQ[result, "Latitude"] && KeyExistsQ[result, "Longitude"],
		lat = Internal`StringToDouble[result["Latitude"]];
   		lon = Internal`StringToDouble[result["Longitude"]];
    	KeyDropFrom[result,{"Latitude","Longitude"}];
    	If[lat != 0 || lon != 0,
    		AssociateTo[result, Rule["Location", GeoPosition[{lat, lon}]]]
    	]
   ];
   result
]

baseEncode[id_] := 
 Module[{baseCount, encoded = "", div, mod, num = FromDigits[id], alphabet = Characters["123456789abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ"]},
  (
   baseCount = Length[alphabet];
   While[num >= baseCount,
    (
     div = num/baseCount;
     mod = (num - (baseCount*Floor[div]));
     encoded = StringJoin[encoded, alphabet[[mod + 1]]];
     num = Floor[div];
     )];
   If[num > 0,
    encoded = StringJoin[encoded, alphabet[[num + 1]]]];
   StringReverse[encoded]
   )]

parseDates[param_] := Block[{startDate, endDate},
	(
   		Switch[param,
    		_String,
    		(
     			startDate = Quiet[Check[DateObject[param], ""]];
     			endDate = startDate;
		    ),
    		_DateObject,
		    (
     			startDate = param;
     			endDate = startDate;
     		),
    		List[_, _],
    		(
     			startDate = param[[1]];
     			endDate = param[[2]];
     			Switch[Head[startDate],
      				String, startDate = Quiet[Check[DateObject[startDate], ""]],
      				DateObject, startDate = startDate
			    ];
     			Switch[Head[endDate],
      				String, endDate = Quiet[Check[DateObject[endDate], ""]],
      				DateObject, endDate = endDate];
		    ),
    		Interval[{_DateObject, _DateObject}],
		    (
     			startDate = param /. Interval[{f_, t_}] :> f;
     			endDate = param /. Interval[{f_, t_}] :> t;
     		),
		    _,
		    (
     			startDate = "";
     			endDate = "";
		    )
		];
   		{startDate, endDate}
   	)]

formatDate[date_] := Block[{d, tz},	
	d = First@StringCases[date, RegularExpression["[0-9]{4}:[0-9]{2}:[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}"]];
	tz = StringDelete[date, d];
	d = StringSplit[d];
	d = (StringReplace[First@d, ":" :> "/"] <> " " <> (Last@d));
	If[StringLength[tz] > 0,
		tz = StringSplit[tz, ":"];
		tz = Internal`StringToDouble[tz[[1]]],
		tz = 0
	];
	DateObject[d, TimeZone -> tz]
]

formatDate2[date_] := Block[{d, tz},	
	d = First@StringCases[date, RegularExpression["[0-9]{4}:[0-9]{2}:[0-9]{2}"]];
	tz = StringDelete[date, d];
	d = StringReplace[d, ":" :> "/"];
	If[StringLength[tz] > 0,
		tz = StringSplit[tz, ":"];
		tz = Internal`StringToDouble[tz[[1]]],
		tz = 0
	];
	DateObject[d, TimeZone -> tz]
]

formatDate3[date_] := Block[{d, tz},	
	d = First@StringCases[date, RegularExpression["[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}"]];
	tz = StringDelete[date, d];
	If[StringLength[tz] > 0,
		tz = StringSplit[tz, ":"];
		tz = Internal`StringToDouble[tz[[1]]],
		tz = 0
	];
	DateObject[d, TimeZone -> tz]
]

formatTime[time_] := Block[{t, tz},
	t = First@StringCases[time, RegularExpression["[0-9]{2}:[0-9]{2}:[0-9]{2}"]];
	tz = StringDelete[time, t];
	If[StringLength[tz] > 0,
		tz = StringSplit[tz, ":"];
		tz = Internal`StringToDouble[tz[[1]]],
		tz = 0
	];
	TimeObject[t, TimeZone -> tz]
]


End[];
 
EndPackage[];