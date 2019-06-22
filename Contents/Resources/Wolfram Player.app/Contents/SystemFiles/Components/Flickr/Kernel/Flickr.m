Begin["Flickr`"]

Get["FlickrFunctions.m"];

(*****************************************Error handling***********************************************)
ServiceExecute::radiuserror="Invalid value specified for radial search, radius must be greater than zero and less than 20 miles (or 32 kilometers)."

Begin["`Private`"]

(******************************* Flickr *************************************)

(* Authentication information *)

flickrdata[]:=
If[TrueQ[OAuthClient`Private`$UseChannelFramework],
{
 	"ServiceName"       -> "Flickr",
 	"OAuthVersion"		-> "1.0a",
 	"RequestEndpoint"   -> "http://www.flickr.com/services/oauth/request_token",
 	"AccessEndpoint"    -> "https://www.flickr.com/services/oauth/access_token",
 	"AuthorizeEndpoint" -> "https://www.flickr.com/services/oauth/authorize",
 	"RedirectURI"       -> "WolframConnectorChannelListen",
 	"Blocking"          -> False,
	"VerifierParsing"   -> "oauth_verifier",
 	"ClientInfo"		-> {"Wolfram","Token"},
	"AuthenticationDialog" -> "WolframConnectorChannel",
 	"Gets"				-> {"PhotoExif","UserGalleries","GalleryInformation","ImageSearch","ImportImage","GalleryImages","UserAlbums","AlbumImages","UserData"},
 	"Posts"				-> {},
 	"RawGets"			-> {"RawPhotoExif","RawUserGalleries","RawFindUserByEmail","RawFindUserByUsername","RawGalleryInfo",
	 							"RawGalleryPhotos","RawPlacesByCoordinates","RawPhotoSearch","RawUserAlbums","RawAlbumImages","RawImageSizes",
	 							"RawPeopleGetInfo","RawTestLogin"},
 	"RawPosts"			-> {},
 	"Information"		-> "A service for exchanging data with a Flickr"
},
{
 	"ServiceName"       -> "Flickr",
 	"OAuthVersion"		-> "1.0a",
 	"RequestEndpoint"   -> "http://www.flickr.com/services/oauth/request_token",
 	"AccessEndpoint"    -> "https://www.flickr.com/services/oauth/access_token",
 	"AuthorizeEndpoint" -> "https://www.flickr.com/services/oauth/authorize",
 	"ClientInfo"		-> {"Wolfram","Token"},
 	"AuthenticationDialog" :> (OAuthClient`tokenOAuthDialog[#, "Flickr"]&),
 	"Gets"				-> {"PhotoExif","UserGalleries","GalleryInformation","ImageSearch","ImportImage","GalleryImages","UserAlbums","AlbumImages","UserData"},
 	"Posts"				-> {},	
 	"RawGets"			-> {"RawPhotoExif","RawUserGalleries","RawFindUserByEmail","RawFindUserByUsername","RawGalleryInfo",
	 							"RawGalleryPhotos","RawPlacesByCoordinates","RawPhotoSearch","RawUserAlbums","RawAlbumImages","RawImageSizes",
	 							"RawPeopleGetInfo","RawTestLogin"},
 	"RawPosts"			-> {},	
 	"Information"		-> "A service for exchanging data with a Flickr"
}]
    
(* Import  *)

flickrimport[json_?StringQ]:= With[{res = Quiet[Developer`ReadRawJSONString[json]]}, 
	(
		If[res["stat"] =!= "fail",
			res,
			Message[ServiceExecute::apierr, res["message"]];
			Throw[$Failed]
		]
	) /; AssociationQ[res]
]

flickrimport[___]:= (Message[ServiceExecute::serror];Throw[$Failed])

flickrnoimport[data_]:= FromCharacterCode[data,"UTF8"]

(* Raw requests *)

flickrdata["RawTestLogin"] := {
	"URL"				-> "https://api.flickr.com/services/rest?method=flickr.test.login",
	"HTTPSMethod"		-> "GET",
	"Parameters"		-> {"format","nojsoncallback","jsoncallback"},
	"RequiredParameters"-> {},
	"ResultsFunction"	-> flickrimport
}

flickrdata["RawPeopleGetInfo"] := {
	"URL"				-> "https://api.flickr.com/services/rest?method=flickr.people.getInfo",
	"HTTPSMethod"		-> "GET",
	"Parameters"		-> {"user_id","format","nojsoncallback","jsoncallback"},
	"RequiredParameters"-> {"user_id"},
	"ResultsFunction"	-> flickrimport
}

flickrdata["RawPhotoExif"] := {
	"URL"				-> "https://api.flickr.com/services/rest?method=flickr.photos.getExif",
	"HTTPSMethod"		-> "GET",
	"Parameters"		-> {"photo_id","secret","format","nojsoncallback","jsoncallback"},
	"RequiredParameters"-> {"photo_id"},
	"ResultsFunction"	-> flickrimport
}

flickrdata["RawUserGalleries"] := {
	"URL"				-> "https://api.flickr.com/services/rest?method=flickr.galleries.getList",
	"HTTPSMethod"		-> "GET",
	"Parameters"		-> {"user_id","per_page","page","primary_photo_extras","format","nojsoncallback","jsoncallback"},
	"RequiredParameters"-> {"user_id"},
	"ResultsFunction"	-> flickrimport
}

flickrdata["RawFindUserByEmail"] := {
	"URL"				-> "https://api.flickr.com/services/rest?method=flickr.people.findByEmail",
	"HTTPSMethod"		-> "GET",
	"Parameters"		-> {"find_email","format","nojsoncallback","jsoncallback"},
	"RequiredParameters"-> {"find_email"},
	"ResultsFunction"	-> flickrimport
}

flickrdata["RawFindUserByUsername"] := {
	"URL"				-> "https://api.flickr.com/services/rest?method=flickr.people.findByUsername",
	"HTTPSMethod"		-> "GET",
	"Parameters"		-> {"username","format","nojsoncallback","jsoncallback"},
	"RequiredParameters"-> {"username"},
	"ResultsFunction"	-> flickrimport
}

flickrdata["RawGalleryInfo"] := {
	"URL"				-> "https://api.flickr.com/services/rest?method=flickr.galleries.getInfo",
	"HTTPSMethod"		-> "GET",
	"Parameters"		-> {"gallery_id","format","nojsoncallback","jsoncallback"},
	"RequiredParameters"-> {"gallery_id"},
	"ResultsFunction"	-> flickrimport
}

flickrdata["RawGalleryPhotos"] := {
	"URL"				-> "https://api.flickr.com/services/rest?method=flickr.galleries.getPhotos",
	"HTTPSMethod"		-> "GET",
	"Parameters"		-> {"gallery_id","extras","per_page","page","format","nojsoncallback","jsoncallback"},
	"RequiredParameters"-> {"gallery_id"},
	"ReturnContentData" -> True,
	"ResultsFunction"	-> flickrnoimport
}

flickrdata["RawImageSizes"] := {
	"URL"				-> "https://api.flickr.com/services/rest?method=flickr.photos.getSizes",
	"HTTPSMethod"		-> "GET",
	"Parameters"		-> {"photo_id","format","nojsoncallback"},
	"RequiredParameters"-> {"photo_id"},
	"ResultsFunction"	-> flickrimport
}

flickrdata["RawPlacesByCoordinates"] := {
	"URL"				-> "https://api.flickr.com/services/rest?method=flickr.places.findByLatLon",
	"HTTPSMethod"		-> "GET",
	(* Values for accuracy: 1,3,6,11,16 *)
	"Parameters"		-> {"lat","lon","accuracy","format","nojsoncallback","jsoncallback"},
	"RequiredParameters"-> {"lat","lon"},
	"ResultsFunction"	-> flickrimport
}
    
flickrdata["RawPhotoSearch"] := {
	"URL"				-> "https://api.flickr.com/services/rest?method=flickr.photos.search",
	"HTTPSMethod"		-> "GET",
	"Parameters"		-> {"user_id","tags","tag_mode","text","min_upload_date","max_upload_date",
							"min_taken_date","max_taken_date","license","sort","privacy_filter","bbox",
							"accuracy","safe_search","content_type","machine_tags","machine_tag_mode",
							"group_id","contacts","woe_id","place_id","media","has_geo","geo_context",
							"lat","lon","radius","radius_units","is_commons","in_gallery","is_getty",
							"extras","per_page","page","format","nojsoncallback","jsoncallback"},
	"RequiredParameters"-> {},
	"ReturnContentData" -> True,
	"ResultsFunction"	-> flickrnoimport
}
    
flickrdata["RawUserAlbums"] := {
	"URL"				-> "https://api.flickr.com/services/rest?method=flickr.photosets.getList",
	"HTTPSMethod"		-> "GET",
	"Parameters"		-> {"user_id","per_page","page","primary_photo_extras","format","nojsoncallback","jsoncallback"},
	"RequiredParameters"-> {},
	"ResultsFunction"	-> flickrimport
}

flickrdata["RawAlbumImages"] := {
	"URL"				-> "https://api.flickr.com/services/rest?method=flickr.photosets.getPhotos",
	"HTTPSMethod"		-> "GET",
	"Parameters"		-> {"photoset_id","extras","privacy_filter","per_page","page","media","format","nojsoncallback","jsoncallback"},
	"RequiredParameters"-> {"photoset_id"},
	"ReturnContentData" -> True,
	"ResultsFunction"	-> flickrnoimport
}

flickrdata[___]:=$Failed

(****** Cooked Properties ******)

flickrcookeddata["UserData", id_, args_?OptionQ] := Block[{params={"format"->"json","nojsoncallback"->"1"},rawdata,userdata,invalidParameters,user,
											userId,loginInfo,userData,boolefields,iconserver,iconfarm},
	
	invalidParameters = Select[Keys[args],!MemberQ[{"User"},#]&]; 
	
	If[Length[invalidParameters]>0,
		Message[ServiceObject::noget,#,"Flickr"]&/@invalidParameters;
		Throw[$Failed]
	];
		
	If[KeyExistsQ[args,"User"],
		user = Lookup[args,"User"];
		userId = getUserId[user,id];
		If[userId === "",
			Message[ServiceExecute::nval,"User","Flickr"];
			Throw[$Failed]
		];
		AppendTo[params,"user_id"->ToString@userId];
	,
		loginInfo = getLoginInfo[id];
		AppendTo[params,"user_id"->loginInfo["user"]["id"]]
	];
	
	rawdata = flickrimport@OAuthClient`rawoauthdata[id,"RawPeopleGetInfo",params];
	
	userData = rawdata["person"];
	iconfarm = userData["iconfarm"]; iconserver = userData["iconserver"];
	userData = KeyDrop[{"mbox_sha1sum","revcontact","revfriend","revfamily","iconfarm","iconserver"}][userData];
	userData = Insert[userData,"BuddyIcon" -> Import[getUserIconUrl[iconfarm,iconserver,userData["nsid"]]],Key["username"]];
	userData = Replace[userData, KeyValuePattern["_content" -> d_] :> d, -2];
	userData = Replace[userData, {asoc_?AssociationQ :> KeyMap[Replace[#, $userCamelFields]&][asoc]}, {0, Infinity}];
	userData["Photos"]["FirstDateTaken"] = Quiet[Check[DateObject[userData["Photos"]["FirstDateTaken"]], Missing["NotAvailable"]]];
	userData["Photos"]["FirstDate"] = Quiet[Check[FromUnixTime[ToExpression[userData["Photos"]["FirstDate"]]], Missing["NotAvailable"]]];
	boolefields = {Key[#]}& /@ Keys[KeyTake[userData, {"IsPro", "CanBuyPro", "IsContact", "IsFriend", "IsFamily", "Ignored", "HasStats", "Expire"}]];
	userData = MapAt[Replace[{0 | "0" -> False, 1 | "1" -> True}], boolefields] @ userData;
	userData = MapAt[Hyperlink, {{Key["PhotosURL"]}, {Key["ProfileURL"]}, {Key["MobileURL"]}}] @ userData;
	Replace[userData, (Null | "")-> Missing["NotAvailable"], {0, Infinity}]
]

flickrcookeddata["PhotoExif", id_, args_?OptionQ] := Block[{params={"format" -> "json", "nojsoncallback" -> "1"},rawdata,data,exifdata,exifkeys,exifrules,invalidParameters},

	invalidParameters = Select[Keys[args],!MemberQ[{"PhotoID"},#]&]; 
	
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Flickr"]&/@invalidParameters;
			Throw[$Failed]
		)];
	
	If[KeyExistsQ[args,"PhotoID"],
		AppendTo[params, "photo_id"->ToString@Lookup[args,"PhotoID"]],
	(
		Message[ServiceExecute::nparam,"PhotoID"];			
		Throw[$Failed]
	)];	
	
	rawdata = flickrimport@OAuthClient`rawoauthdata[id,"RawPhotoExif",params];
	
	data = rawdata["photo"];
	exifdata = Association[Rule[#["tag"], #["raw"]["_content"]]& /@ data["exif"]];
	data = KeyMap[Replace[#,{"id" -> "ID", "secret" -> "Secret", "server" -> "Server","farm" -> "Farm", "camera" -> "Camera", "exif" -> "Exif"}]&,data];
	exifkeys = Cases[Keys[exifdata],
		Alternatives["ModifyDate", "DateTimeOriginal", "CreateDate", "DateCreated", "TimeCreated", "DigitalCreationDate", "DigitalCreationTime", "MetadataDate", "HistoryWhen"]];	
	exifrules = FilterRules[{
		Rule["ModifyDate", Quiet[Check[formatDate[exifdata["ModifyDate"]], exifdata["ModifyDate"]]]],
		Rule["DateTimeOriginal", Quiet[Check[formatDate[exifdata["DateTimeOriginal"]], exifdata["DateTimeOriginal"]]]],
		Rule["CreateDate", Quiet[Check[formatDate[exifdata["CreateDate"]], exifdata["CreateDate"]]]],
		Rule["DateCreated", Quiet[Check[formatDate2[exifdata["DateCreated"]], exifdata["DateCreated"]]]],
		Rule["TimeCreated", Quiet[Check[formatTime[exifdata["TimeCreated"]], exifdata["TimeCreated"]]]],
		Rule["DigitalCreationDate", Quiet[Check[formatDate2[exifdata["DigitalCreationDate"]], exifdata["DigitalCreationDate"]]]],
		Rule["DigitalCreationTime", Quiet[Check[formatTime[exifdata["DigitalCreationTime"]], exifdata["DigitalCreationTime"]]]],
		Rule["MetadataDate", Quiet[Check[formatDate[exifdata["MetadataDate"]], exifdata["MetadataDate"]]]],
		Rule["HistoryWhen", Quiet[Check[formatDate[exifdata["HistoryWhen"]], exifdata["HistoryWhen"]]]]
		},exifkeys];
	AssociateTo[exifdata,exifrules];
	AppendTo[data,"Exif"->exifdata]
]

flickrcookeddata["UserGalleries", id_, args_?OptionQ] := Block[{userId,params={"format"->"json","nojsoncallback"->"1"},showq,rawdata,galleriesdata,
															invalidParameters,user,loginInfo,limit},

	invalidParameters = Select[Keys[args],!MemberQ[{"User","MaxItems",MaxItems,"ShowPrimaryPhoto"},#]&]; 
	
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Flickr"]&/@invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"User"],
	(
		user = Lookup[args,"User"];
		userId = getUserId[user,id];
		If[userId=="",
		(
			Message[ServiceExecute::nval,"User","Flickr"];
			Throw[$Failed]
		)];
		AppendTo[params,"user_id"->ToString@userId];
	),
	(
		loginInfo = getLoginInfo[id];
		AppendTo[params,"user_id"->loginInfo["user"]["id"]]
	)];
	
	If[Xor[KeyExistsQ[args,"MaxItems"],KeyExistsQ[args,MaxItems]],
		limit = First@DeleteMissing[Lookup[args, {"MaxItems", MaxItems}]];
		If[!(IntegerQ[limit] && limit>0),
		(	
			Message[ServiceExecute::nval,"MaxItems","Flickr"];
			Throw[$Failed]
		)];
		AppendTo[params,"per_page"->ToString@limit]
	];

	If[KeyExistsQ[args,"ShowPrimaryPhoto"],
		showq = Lookup[args,"ShowPrimaryPhoto"];
		If[!BooleanQ[showq],
			Message[ServiceExecute::nval,"ShowPrimaryPhoto","Flickr"];
			Throw[$Failed]
		],
		showq = False
	];
		
	rawdata = flickrimport@OAuthClient`rawoauthdata[id,"RawUserGalleries",params];
	
	galleriesdata = rawdata["galleries"]["gallery"];

	Switch[showq,
		True,
			Dataset[<|"PrimaryPhoto"->
			Image[Import[getphotourl[#["primary_photo_farm"],#["primary_photo_server"],#["primary_photo_id"],#["primary_photo_secret"]]],
				MetaInformation->{"Source"->#["url"]}],
			"ID"->#["id"],"Title"->#["title"]["_content"],
			"DateCreate"->FromUnixTime[ToExpression[#["date_create"]]],
			"CountPhotos"->FromDigits[#["count_photos"]],
			"URL"-> #["url"]|>&/@galleriesdata],
		False,
			Dataset[<|"ID"->#["id"],"Title"->#["title"]["_content"],
			"DateCreate"->FromUnixTime[ToExpression[#["date_create"]]],
			"CountPhotos"->FromDigits[#["count_photos"]],
			"URL"-> #["url"]|>&/@galleriesdata]
	]
]


flickrcookeddata["GalleryInformation", id_, args_?OptionQ] := Block[{params={"format" -> "json", "nojsoncallback" -> "1"},rawdata,gallerydata,primaryphotourl,
	invalidParameters},
	invalidParameters = Select[Keys[args],!MemberQ[{"GalleryID"},#]&]; 
	
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Flickr"]&/@invalidParameters;
			Throw[$Failed]
		)];
	
	If[KeyExistsQ[args,"GalleryID"],
		AppendTo[params, "gallery_id"->ToString@Lookup[args,"GalleryID"]],
	(
		Message[ServiceExecute::nparam,"GalleryID"];			
		Throw[$Failed]
	)];

	rawdata = flickrimport@OAuthClient`rawoauthdata[id,"RawGalleryInfo",params];

	gallerydata = rawdata["gallery"];
	
	primaryphotourl = getphotourl[#["primary_photo_farm"],#["primary_photo_server"],#["primary_photo_id"],#["primary_photo_secret"]]&@gallerydata;
	
	<|"ID"->#["id"],"Title"->#["title"]["_content"],"Description"->#["description"]["_content"],"Username"->#["username"],"Owner"->#["owner"],
	"DateCreate"->FromUnixTime[ToExpression[#["date_create"]]],"DateUpdate"->FromUnixTime[ToExpression[#["date_update"]]],
	"CountViews"->FromDigits[#["count_views"]],"CountPhotos"->FromDigits[#["count_photos"]],"CountVideos"->FromDigits[#["count_videos"]],"CountComments"->FromDigits[#["count_comments"]],
	"IconFarm"->#["iconfarm"],"IconServer"->#["iconserver"],"PrimaryPhotoID"->#["primary_photo_id"],"PrimaryPhoto"->primaryphotourl,
	"PrimaryPhotoServer"->#["primary_photo_server"],"PrimaryPhotoFarm"->#["primary_photo_farm"],"PrimaryPhotoSecret"->#["primary_photo_secret"],
	"URL"-> #["url"]|>&@gallerydata

]

flickrcookeddata["ImageSearch", id_, args_?OptionQ] := Block[{invalidParameters,keywords,description,dateTaken,dateUploaded,startDate,endDate,
												location,coordinates,latitude,longitude,placesRawData,placeId,format,startIndex,startPage,remainder,calls,
												element,photosInfo,rawparams={},raw,rawdata,result,outputData,size,orgFormat,maxPerPage=30,limit,accuracy,
												temporal,radius,radiusUnits,bbox,user,userinfo,userid,sort,sortParam,point,progress=0},

	invalidParameters = Select[Keys[args],!MemberQ[{"Keywords","Description","User","SortBy","DateTaken","DateUploaded","Location",
													"Elements","MaxItems",MaxItems,"StartIndex","Format","ImageSize"},#]&]; 
	
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Flickr"]&/@invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"Keywords"],

		keywords = Lookup[args,"Keywords"];
		
		If[StringQ[keywords],keywords = {keywords}];
		
		If[Length[keywords] == 1 && MatchQ[Head[keywords[[1]]], Alternatives],
			AppendTo[rawparams,Rule["tag_mode","any"]];
			keywords = List @@ keywords[[1]]
			,
			AppendTo[rawparams,Rule["tag_mode","all"]];
		];
		AppendTo[rawparams,Rule["tags",StringJoin[Riffle[keywords, ","]]]];
	];
	
	If[KeyExistsQ[args,"Description"],
	(
		description = Lookup[args,"Description"];
		AppendTo[rawparams,Rule["text",description]];
	)];
	
	If[KeyExistsQ[args,"User"],
	(
		user = Lookup[args,"User"];
		userid = getUserId[user,id];
		If[userid=="",
		(
			Message[ServiceExecute::nval,"User","Flickr"];
			Throw[$Failed]
		)];
		AppendTo[rawparams,"user_id"->ToString@userid];	
	)];

	If[KeyExistsQ[args,"SortBy"],
	(
		sort = Lookup[args,"SortBy"];
		If[StringQ[sort],
		(
			If[sort == "Relevance",
			(
				AppendTo[rawparams,Rule["sort","relevance"]];
			),
			(
				(* Default descending *)
				sort = {sort,"Descending"};
			)]
		)];
		If[MatchQ[sort, {_String, _String}],
		(
			sortParam = "";
			Switch[sort[[1]],
				"DatePosted",
				(
					If[sort[[2]] == "Ascending", sortParam = "date-posted-asc"];
					If[sort[[2]] == "Descending", sortParam = "date-posted-desc"];
				),
				"DateTaken",
				(
					If[sort[[2]] == "Ascending", sortParam = "date-taken-asc"];
					If[sort[[2]] == "Descending", sortParam = "date-taken-desc"];
				),
				"Interestingness",
				(
					If[sort[[2]] == "Ascending", sortParam = "interestingness-asc"];
					If[sort[[2]] == "Descending", sortParam = "interestingness-asc"];
				)];
				
				If[sortParam != "",
					AppendTo[rawparams,Rule["sort",sortParam]];];
		)];				
	)];
	
	If[KeyExistsQ[args,"DateTaken"],
	(
		dateTaken = Lookup[args,"DateTaken"];
		
		{startDate,endDate} = parseDates[dateTaken];
		
		If[!DateObjectQ[startDate],
		(
			Message[ServiceExecute::nval,"DateTaken","Flickr"];	
			Throw[$Failed]
		)];
		
		If[!DateObjectQ[endDate],
		(
			Message[ServiceExecute::nval,"DateTaken","Flickr"];	
			Throw[$Failed]
		)];
		
		startDate = ToString[UnixTime[startDate]];
		endDate = ToString[UnixTime[endDate]];
				
		AppendTo[rawparams,Rule["min_taken_date",startDate]];
		AppendTo[rawparams,Rule["max_taken_date",endDate]];
		
	)];
	
	If[KeyExistsQ[args,"DateUploaded"],
	(
		dateUploaded = Lookup[args,"DateUploaded"];
		
		{startDate,endDate} = parseDates[dateUploaded];
		
		If[!DateObjectQ[startDate],
		(
			Message[ServiceExecute::nval,"DateUploaded","Flickr"];	
			Throw[$Failed]
		)];
		
		If[!DateObjectQ[endDate],
		(
			Message[ServiceExecute::nval,"DateUploaded","Flickr"];	
			Throw[$Failed]
		)];
		
		startDate = ToString[UnixTime[startDate]];
		endDate = ToString[UnixTime[endDate]];
				
		AppendTo[rawparams,Rule["min_upload_date",startDate]];
		AppendTo[rawparams,Rule["max_upload_date",endDate]];
		
	)];
	
	If[KeyExistsQ[args,"Location"],
	(
		location = Lookup[args,"Location"];

		(* this handles the case where the user gives a GeoPosition representation for more than one point e.g. polygons *)
		If[MatchQ[Head[location],GeoPosition] && MatchQ[Head[QuantityMagnitude[Latitude[location], "AngularDegrees"]],List],
			location=GeoBoundingBox[location]]; 

		Switch[Head[location],
			GeoPosition, (* radial search, default radius 5km *)
			(
				latitude = QuantityMagnitude[Latitude[location], "AngularDegrees"] //ToString;
				longitude = QuantityMagnitude[Longitude[location], "AngularDegrees"] //ToString;
				
				rawparams = Join[rawparams,{Rule["lat",latitude],Rule["lon",longitude]}];
			),
			Entity,
			(
				Switch[EntityTypeName[location],
					"Country",
					(
						latitude = QuantityMagnitude[Latitude[location], "AngularDegrees"] //ToString;
						longitude = QuantityMagnitude[Longitude[location], "AngularDegrees"] //ToString;
						accuracy = "1";
						placesRawData = ImportString[ToString@OAuthClient`rawoauthdata[id,"RawPlacesByCoordinates",{"lat" -> latitude, "lon" -> longitude, "accuracy" -> accuracy, "format" -> "json", "nojsoncallback" -> "1"}],"JSON"];
						placeId = "place_id" /. ("place" /. ("places" /. placesRawData))[[1]];
						
						rawparams = Append[rawparams,Rule["place_id",placeId]];
					),
					"City",
					(
						latitude = QuantityMagnitude[Latitude[location], "AngularDegrees"] //ToString;
						longitude = QuantityMagnitude[Longitude[location], "AngularDegrees"] //ToString;
						accuracy = "11";
						placesRawData = ImportString[ToString@OAuthClient`rawoauthdata[id,"RawPlacesByCoordinates",{"lat" -> latitude, "lon" -> longitude, "accuracy" -> accuracy, "format" -> "json", "nojsoncallback" -> "1"}],"JSON"];
						placeId = "place_id" /. ("place" /. ("places" /. placesRawData))[[1]];
						
						rawparams = Append[rawparams,Rule["place_id",placeId]];
					),
					_,
					(
						coordinates = LatitudeLongitude[location];
						If[MatchQ[coordinates,List[_Quantity,_Quantity]],
						(
							latitude = QuantityMagnitude[coordinates[[1]]] // ToString;
							longitude = QuantityMagnitude[coordinates[[2]]] // ToString;

							rawparams = Join[rawparams,{Rule["lat",latitude],Rule["lon",longitude]}];
						),
						(
							Message[ServiceExecute::nval,"Location","Flickr"];	
							Throw[$Failed]
						)]
					)
				]
			),
			GeoDisk,
			(
				Switch[location,
					GeoDisk[],
					(
						point = $GeoLocation;
						radius = 5000;
					),
					GeoDisk[_],
					(	
						point = location[[1]];
						radius = 5000;
					),
					GeoDisk[_,_,___],
					(
						point = location[[1]];
						radius = location[[2]];						
					)
				];
				
				latitude = QuantityMagnitude[Latitude[point], "AngularDegrees"] //ToString;
				longitude = QuantityMagnitude[Longitude[point], "AngularDegrees"] //ToString;
				
				If[Internal`RealValuedNumericQ[radius], 
					radius = radius / 1000, (* GeoDisk assumes that the quantity representing the radius is in meters *)
					radius = QuantityMagnitude[radius, "Kilometers"]
				];
				
				If[radius>=0 && radius<=32,
				(
					radius = ToString[radius];
					rawparams = Join[rawparams,{Rule["lat",latitude],Rule["lon",longitude],Rule["radius",radius]}];
				),
				(
					Message[ServiceExecute::radiuserror];	
					(*Message[ServiceExecute::nval,"Location","Flickr"];*)
					Throw[$Failed]
				)]
				
				
			),
			List,
			(
				If[MatchQ[location, List[GeoPosition[___], GeoPosition[___]]],
				(
					bbox = StringJoin[Riffle[ToString[#[[1]]] & /@ {Longitude[location[[1]]],Latitude[location[[1]]],
								Longitude[location[[2]]], Latitude[location[[2]]]},","]];
								
					rawparams = Append[rawparams,Rule["bbox",bbox]];
				)]				
			),
			_, (* unrecognized Location specification *)
			(
				Message[ServiceExecute::nval,"Location","Flickr"];	
				Throw[$Failed]
			)					
		]
	)];
	
	If[KeyExistsQ[args,"Elements"],
		element = Lookup[args,"Elements"];
		If[StringQ[element],element = {element}];
		
		Switch[element,
			{("Images"|"Data"|"FullData"|"ImageLinks"|"LinkedThumbnails")..},
				element,
			_,
				Message[ServiceExecute::nval,"Elements","Flickr"];
				Throw[$Failed]
			]
		,
		element = {"Data"}];

	If[MemberQ[element,("Data"|"FullData")] && KeyExistsQ[args,"Format"],
		format = Lookup[args,"Format"];
		Switch[format,
			("JSON"|"XML"|"Association"|"Dataset"),
				format,
			_,
				Message[ServiceExecute::nval,"Format","Flickr"];
				Throw[$Failed]
			],
		format = "Dataset"
	];
	
	If[format =!= "XML", rawparams = Join[rawparams,{"format" -> "json", "nojsoncallback" -> "1"}]];

	If[Xor[KeyExistsQ[args,"MaxItems"],KeyExistsQ[args,MaxItems]],
		limit = First@DeleteMissing[Lookup[args, {"MaxItems", MaxItems}]];
		If[!(IntegerQ[limit] && limit>0),
		(	
			Message[ServiceExecute::nval,"MaxItems","Flickr"];
			Throw[$Failed]
		)];
			(* the max value allowed is 30 *)
			limit = Min[limit,maxPerPage],
		limit = maxPerPage;
	];

	If[KeyExistsQ[args,"StartIndex"],
		(
			startIndex = Lookup[args,"StartIndex"];
			If[!IntegerQ[startIndex],
			(	
				Message[ServiceExecute::nval,"StartIndex","Flickr"];
				Throw[$Failed]
			)];
		),
		startIndex = 1
	];

	rawparams = Join[rawparams,{"extras" -> "description,license,date_upload,date_taken,owner_name,original_format,last_update,geo,tags,machine_tags,views"}];
	
	startPage = 1 + Quotient[startIndex,maxPerPage,1];
	remainder = Mod[startIndex,maxPerPage,1];

	calls = Ceiling[(remainder + limit) / maxPerPage*1.];

	photosInfo = {};raw={};
	PrintTemporary[ProgressIndicator[Dynamic[progress], {0, calls}]];
	rawparams = Join[rawparams,{"per_page" -> ToString[maxPerPage],"page" -> ToString[1]}];
	(
		rawparams = ReplaceAll[rawparams,{Rule["per_page",_]->Rule["per_page",ToString[maxPerPage]],Rule["page",_]->Rule["page",ToString[startPage + #]]}];
		If[format=="XML",
			rawdata = FromCharacterCode[OAuthClient`rawoauthdata[id,"RawPhotoSearch",rawparams],"UTF-8"];
			raw = Append[raw,rawdata],
			rawdata = flickrimport[FromCharacterCode[OAuthClient`rawoauthdata[id,"RawPhotoSearch",rawparams],"UTF-8"]];	
			photosInfo = Join[photosInfo,rawdata["photos"]["photo"]]
		];
		progress = progress + 1;
		
	)& /@ Range[0,calls-1];

	(* select photos here *)
	photosInfo = Take[photosInfo,{remainder,UpTo[remainder+limit-1]}];
	
	result = Rule[#,Switch[#,
				"Images", 
				(
					If[KeyExistsQ[args,"ImageSize"],
						size = Lookup[args,"ImageSize"];
						If[!MatchQ[size,"Small" | "Medium" | "Large" | "Thumbnail" | "Original"],
							Message[ServiceExecute::nval,"ImageSize","Flickr"];
							Throw[$Failed]
						];
						,
						size  = "Medium"
					];
					size = mapImageSizes[size];
					getPhoto[#, size, True]& /@ photosInfo
				),
				"Data",
				(
					Switch[format,
						"JSON",
						photosInfo,
						"XML",
						raw,
						"Association",
						<|"Thumbnail" -> getPhoto[#, "t", True], "Keys"-> <|"ID" -> #["id"], "Farm" -> #["farm"], "Server" -> #["server"], "Secret" -> #["secret"]|>,
						"Owner" -> #["owner"], "Title" -> #["title"]|> & /@ photosInfo,
						_,
						outputData = <|"Thumbnail" -> getPhoto[#, "t", True], "Keys"-> <|"ID" -> #["id"], "Farm" -> #["farm"], "Server" -> #["server"], "Secret" -> #["secret"]|>,
						"Owner" -> #["owner"], "Title" -> #["title"]|> & /@ photosInfo;
						Dataset[outputData]
					]
				),
				"FullData",
				(
					Switch[format,
						"JSON",
						photosInfo,
						"XML",
						raw,
						"Association",
						outputData = Block[{var = #},
							Association@RotateRight[
								FilterRules[
									KeyMap[Replace[#,{"owner" -> "Owner", "ispublic" -> "IsPublic", "title" -> "Title", "license" -> "License",
										"ownername" -> "Ownername", "lastupdate" -> "LastUpdate", "latitude" -> "Latitude", "longitude" -> "Longitude", "tags" -> "Tags", "machine_tags" -> "MachineTags", "views" -> "Views", "place_id" -> "PlaceID"}]&,
										AssociateTo[var, {"description" -> var["description"]["_content"],
										"OriginalFormat"->ToUpperCase[Lookup[var,"originalformat","JPG"]], "DateTaken"->Quiet[Check[formatDate3[var["datetaken"]], var["datetaken"]]],
										"DateUploaded"->FromUnixTime[ToExpression[var["dateupload"]]],"LastUpdate"->FromUnixTime[ToExpression[var["lastupdate"]]],
										"Keys" -> <|"ID" -> var["id"], "Farm" -> var["farm"], "Server" -> var["server"], "Secret" -> var["secret"]|>,
										"Thumbnail" -> getPhoto[var, "t", True]}
										]
									],
								{"DateTaken", "DateUpload", "Description", "IsPublic", "Keys", "LastUpdate", "Latitude", "License", "Longitude", "MachineTags", "OriginalFormat", "Owner", "Ownername", "PlaceID", "Tags", "Thumbnail", "Title", "Views"}
								]
							]
						] & /@ photosInfo;
						formatImageFullData/@outputData,
						_,
						outputData = Block[{var = #},
							Association@RotateRight[
								FilterRules[
									KeyMap[Replace[#,{"owner" -> "Owner", "ispublic" -> "IsPublic", "title" -> "Title", "license" -> "License",
										"ownername" -> "Ownername", "lastupdate" -> "LastUpdate", "latitude" -> "Latitude", "longitude" -> "Longitude", "tags" -> "Tags", "machine_tags" -> "MachineTags", "views" -> "Views", "place_id" -> "PlaceID"}]&,
										AssociateTo[var, {"description" -> var["description"]["_content"],
										"OriginalFormat"->ToUpperCase[Lookup[var,"originalformat","JPG"]], "DateTaken"->Quiet[Check[formatDate3[var["datetaken"]], var["datetaken"]]],
										"DateUploaded"->FromUnixTime[ToExpression[var["dateupload"]]],"LastUpdate"->FromUnixTime[ToExpression[var["lastupdate"]]],
										"Keys" -> <|"ID" -> var["id"], "Farm" -> var["farm"], "Server" -> var["server"], "Secret" -> var["secret"]|>,
										"Thumbnail" -> getPhoto[var, "t", True]}
										]
									],
								{"DateTaken", "DateUpload", "Description", "IsPublic", "Keys", "LastUpdate", "Latitude", "License", "Longitude", "MachineTags", "OriginalFormat", "Owner", "Ownername", "PlaceID", "Tags", "Thumbnail", "Title", "Views"}
								]
							]
						] & /@ photosInfo;
						Dataset[formatImageFullData/@outputData]
					]
				),
				"ImageLinks",
				(
					If[KeyExistsQ[args,"ImageSize"],
						size = Lookup[args,"ImageSize"];
						If[!MatchQ[size,"Small" | "Medium" | "Large" | "Thumbnail" | "Original"],
							Message[ServiceExecute::nval,"ImageSize","Flickr"];
							Throw[$Failed]
						];
						,
						size  = "Medium"
					];
					size = mapImageSizes[size];
					Hyperlink[getPhoto[#, size, False]]& /@ photosInfo
				),
				"LinkedThumbnails",
				(
					Hyperlink[getPhoto[#, "t", True],getPhotoMainURL[#]]& /@ photosInfo
				),
				_,
				(
					Message[ServiceExecute::nval,"Elements","Flickr"];
					Throw[$Failed]
				)
			]]& /@ element;
	If[Length[result] == 1, result[[1,2]],result]	

]

flickrcookeddata["ImportImage", id_, args_?OptionQ] := Block[{keys,size,invalidParameters},
	invalidParameters = Select[Keys[args],!MemberQ[{"Keys","ImageSize"},#]&]; 
	
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Flickr"]&/@invalidParameters;
			Throw[$Failed]
		)];	
	
	
	If[KeyExistsQ[args,"Keys"],
	(
		keys = Lookup[args,"Keys"];
		If[!MatchQ[keys,<|"ID" -> _, "Farm" -> _, "Server" -> _,"Secret" -> _|>],
			Message[ServiceExecute::nval,"Keys","Flickr"];
			Throw[$Failed]
		];
		keys = KeyMap[ToLowerCase,keys];
		If[KeyExistsQ[args,"ImageSize"],
			size = Lookup[args,"ImageSize"];
			If[!MatchQ[size,"Small" | "Medium" | "Large" | "Thumbnail" | "Original"],
				Message[ServiceExecute::nval,"ImageSize","Flickr"];
				Throw[$Failed]
			];
			,
			size  = "Thumbnail"
		];
		size = mapImageSizes[size];
		If[size =!= "o",
			getPhoto[keys, size, True],
			getOriginalPhoto[keys, True]
		]
	),
	(
		Message[ServiceExecute::nparam,"Keys"];			
		Throw[$Failed]
	)]
		
]

flickrcookeddata["GalleryImages", id_, args_?OptionQ] := Block[{galleryId, element, format, rawparams={}, rawdata, photosInfo, result, size, 
														outputData, orgFormat, invalidParameters},
	invalidParameters = Select[Keys[args],!MemberQ[{"GalleryID","Elements","Format","ImageSize"},#]&]; 
	
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Flickr"]&/@invalidParameters;
			Throw[$Failed]
		)];	
	
	If[KeyExistsQ[args,"GalleryID"],
		galleryId = Lookup[args,"GalleryID"];
		Switch[galleryId,
			(_String|_Integer),
				rawparams = Append[rawparams, Rule["gallery_id",ToString@galleryId]],
			 _,
				Message[ServiceExecute::nval,"GalleryID","Flickr"];
				Throw[$Failed]
		],
			Message[ServiceExecute::nparam,"GalleryID"];			
			Throw[$Failed]
	];
	
	If[KeyExistsQ[args,"Elements"],
		element = Lookup[args,"Elements"];
		If[StringQ[element],element = {element}];
		
		Switch[element,
			{("Images"|"Data"|"FullData"|"ImageLinks"|"LinkedThumbnails")..},
				element,
			_,
				Message[ServiceExecute::nval,"Elements","Flickr"];
				Throw[$Failed]
			]
		,
		element = {"Data"}];

	If[MemberQ[element,("Data"|"FullData")] && KeyExistsQ[args,"Format"],
		format = Lookup[args,"Format"];
		Switch[format,
			("JSON"|"XML"|"Association"|"Dataset"),
				format,
			_,
				Message[ServiceExecute::nval,"Format","Flickr"];
				Throw[$Failed]
			],
		format = "Dataset"
	];
	
	If[format =!= "XML", rawparams = Join[rawparams,{"format" -> "json", "nojsoncallback" -> "1"}]];
	
	rawparams = Join[rawparams,{"extras" -> "description,license,date_upload,date_taken,owner_name,original_format,last_update,geo,tags,machine_tags,views"}];

	If[format=="XML",
		rawdata = FromCharacterCode[OAuthClient`rawoauthdata[id,"RawGalleryPhotos",rawparams],"UTF-8"],
		rawdata = flickrimport[FromCharacterCode[OAuthClient`rawoauthdata[id,"RawGalleryPhotos",rawparams],"UTF-8"]];
	];
	
	photosInfo = rawdata["photos"]["photo"];

	result = Rule[#,Switch[#,
				"Images", 
				(
					If[KeyExistsQ[args,"ImageSize"],
						size = Lookup[args,"ImageSize"];
						If[!MatchQ[size,"Small" | "Medium" | "Large" | "Thumbnail" | "Original"],
							Message[ServiceExecute::nval,"ImageSize","Flickr"];
							Throw[$Failed]
						];
						,
						size  = "Medium"
					];
					size = mapImageSizes[size];	
					getPhoto[#, size, True]& /@ photosInfo
				),
				"Data",
				(
					Switch[format,
						"JSON",
						photosInfo,
						"XML",
						rawdata,
						"Association",
						<|"Thumbnail" -> getPhoto[#, "t", True], "Keys"-> <|"ID" -> #["id"], "Farm" -> #["farm"], "Server" -> #["server"], "Secret" -> #["secret"]|>,
						"Owner" -> #["owner"], "Title" -> #["title"]|> & /@ photosInfo,
						_,
						outputData = <|"Thumbnail" -> getPhoto[#, "t", True], "Keys"-> <|"ID" -> #["id"], "Farm" -> #["farm"], "Server" -> #["server"], "Secret" -> #["secret"]|>,
						"Owner" -> #["owner"], "Title" -> #["title"]|> & /@ photosInfo;
						Dataset[outputData]
					]
				),
				"FullData",
				(
					Switch[format,
						"JSON",
						rawdata,
						"XML",
						rawdata,
						"Association",
						outputData = Block[{var = #},
							Association@RotateRight[
								FilterRules[
									KeyMap[Replace[#,{"owner" -> "Owner", "ispublic" -> "IsPublic", "title" -> "Title", "license" -> "License",
										"ownername" -> "Ownername", "lastupdate" -> "LastUpdate", "latitude" -> "Latitude", "longitude" -> "Longitude", "tags" -> "Tags", "machine_tags" -> "MachineTags", "views" -> "Views", "place_id" -> "PlaceID"}]&,
										AssociateTo[var, {"description" -> var["description"]["_content"],
										"OriginalFormat"->ToUpperCase[Lookup[var,"originalformat","JPG"]], "DateTaken"->Quiet[Check[formatDate3[var["datetaken"]], var["datetaken"]]],
										"DateUploaded"->FromUnixTime[ToExpression[var["dateupload"]]],"LastUpdate"->FromUnixTime[ToExpression[var["lastupdate"]]],
										"Keys" -> <|"ID" -> var["id"], "Farm" -> var["farm"], "Server" -> var["server"], "Secret" -> var["secret"]|>,
										"Thumbnail" -> getPhoto[var, "t", True]}
										]
									],
								{"DateTaken", "DateUpload", "Description", "IsPublic", "Keys", "LastUpdate", "Latitude", "License", "Longitude", "MachineTags", "OriginalFormat", "Owner", "Ownername", "PlaceID", "Tags", "Thumbnail", "Title", "Views"}
								]
							]
						] & /@ photosInfo;
						formatImageFullData/@outputData,
						_,
						outputData = Block[{var = #},
							Association@RotateRight[
								FilterRules[
									KeyMap[Replace[#,{"owner" -> "Owner", "ispublic" -> "IsPublic", "title" -> "Title", "license" -> "License",
										"ownername" -> "Ownername", "lastupdate" -> "LastUpdate", "latitude" -> "Latitude", "longitude" -> "Longitude", "tags" -> "Tags", "machine_tags" -> "MachineTags", "views" -> "Views", "place_id" -> "PlaceID"}]&,
										AssociateTo[var, {"description" -> var["description"]["_content"],
										"OriginalFormat"->ToUpperCase[Lookup[var,"originalformat","JPG"]], "DateTaken"->Quiet[Check[formatDate3[var["datetaken"]], var["datetaken"]]],
										"DateUploaded"->FromUnixTime[ToExpression[var["dateupload"]]],"LastUpdate"->FromUnixTime[ToExpression[var["lastupdate"]]],
										"Keys" -> <|"ID" -> var["id"], "Farm" -> var["farm"], "Server" -> var["server"], "Secret" -> var["secret"]|>,
										"Thumbnail" -> getPhoto[var, "t", True]}
										]
									],
								{"DateTaken", "DateUpload", "Description", "IsPublic", "Keys", "LastUpdate", "Latitude", "License", "Longitude", "MachineTags", "OriginalFormat", "Owner", "Ownername", "PlaceID", "Tags", "Thumbnail", "Title", "Views"}
								]
							]
						] & /@ photosInfo;
						Dataset[formatImageFullData/@outputData]
					]
				),
				"ImageLinks",
				(
					If[KeyExistsQ[args,"ImageSize"],
						size = Lookup[args,"ImageSize"];
						If[!MatchQ[size,"Small" | "Medium" | "Large" | "Thumbnail" | "Original"],
							Message[ServiceExecute::nval,"ImageSize","Flickr"];
							Throw[$Failed]
						];
						,
						size  = "Medium"
					];
					size = mapImageSizes[size];
					Hyperlink[getPhoto[#, size, False]]& /@ photosInfo
				),
				"LinkedThumbnails",
				(
					Hyperlink[getPhoto[#, "t", True],getPhotoMainURL[#]]& /@ photosInfo
				),
				_,
				(
					Message[ServiceExecute::nval,"Elements","Flickr"];
					Throw[$Failed]
				)
			]]& /@ element;
	If[Length[result] == 1, result[[1,2]],result]		
]

flickrcookeddata["UserAlbums", id_, args_?OptionQ] := Block[{userId,params={"format"->"json","nojsoncallback"->"1","page"->"1"},limit,showq,rawdata,albumsdata,user,invalidParameters,loginInfo},
	invalidParameters = Select[Keys[args],!MemberQ[{"User","MaxItems",MaxItems,"ShowPrimaryPhoto"},#]&]; 
	
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Flickr"]&/@invalidParameters;
			Throw[$Failed]
		)];	

	If[KeyExistsQ[args,"User"],
	(
		user = Lookup[args,"User"];
		userId = getUserId[user,id];
		If[userId=="",
		(
			Message[ServiceExecute::nval,"User","Flickr"];
			Throw[$Failed]
		)];
		AppendTo[params,"user_id"->ToString@userId];
	),
	(
		loginInfo = getLoginInfo[id];
		AppendTo[params,"user_id"->loginInfo["user"]["id"]]
	)];

	If[Xor[KeyExistsQ[args,"MaxItems"],KeyExistsQ[args,MaxItems]],
		limit = First@DeleteMissing[Lookup[args, {"MaxItems", MaxItems}]];
		If[!(IntegerQ[limit] && limit>0),
		(	
			Message[ServiceExecute::nval,"MaxItems","Flickr"];
			Throw[$Failed]
		)];
		AppendTo[params,"per_page"->ToString@limit]
	];

	If[KeyExistsQ[args,"ShowPrimaryPhoto"],
		showq = Lookup[args,"ShowPrimaryPhoto"];
		If[!BooleanQ[showq],
			Message[ServiceExecute::nval,"ShowPrimaryPhoto","Flickr"];
			Throw[$Failed]
		],
		showq = False
	];
		
	rawdata = flickrimport@OAuthClient`rawoauthdata[id,"RawUserAlbums",params];

	albumsdata = rawdata["photosets"]["photoset"];

	Switch[showq,
		True,
			Dataset[<|"PrimaryPhoto"-> Image[Import[getphotourl[#["farm"], #["server"], #["primary"], #["secret"]]],
				MetaInformation->{"Source"->Hyperlink[StringJoin["https://www.flickr.com/photos/",ToString[userId],"/sets/",#["id"]]]}],
			"ID"->#["id"],"Title"->#["title"]["_content"],
			"Description"->#["description"]["_content"],
			"Photos"->#["photos"],
			"Videos"->#["videos"],
			"DateCreate"->FromUnixTime[ToExpression[#["date_create"]]]|>&/@albumsdata],
		False,
			Dataset[<|"ID"->#["id"],"Title"->#["title"]["_content"],
			"Description"->#["description"]["_content"],
			"Photos"->#["photos"],
			"Videos"->#["videos"],
			"DateCreate"->FromUnixTime[ToExpression[#["date_create"]]]|>&/@albumsdata]
	]
]

flickrcookeddata["AlbumImages", id_, args_?OptionQ] := Block[{photosetId, element, format, rawparams={}, rawdata, photosInfo, result, 
														size, outputData, orgFormat, invalidParameters, limit},
	invalidParameters = Select[Keys[args],!MemberQ[{"AlbumID","Elements","Format","ImageSize","MaxItems",MaxItems},#]&]; 
	
	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"Flickr"]&/@invalidParameters;
			Throw[$Failed]
		)];
	
	If[KeyExistsQ[args,"AlbumID"],
		rawparams = Append[rawparams, Rule["photoset_id",Lookup[args,"AlbumID"]]],
		(
			Message[ServiceExecute::nparam,"AlbumID"];			
			Throw[$Failed]
	)];		
	
	If[Xor[KeyExistsQ[args,"MaxItems"],KeyExistsQ[args,MaxItems]],
		limit = First@DeleteMissing[Lookup[args, {"MaxItems", MaxItems}]];
		If[!(IntegerQ[limit] && limit>0),
		(	
			Message[ServiceExecute::nval,"MaxItems","Flickr"];
			Throw[$Failed]
		)];
		AppendTo[rawparams,"per_page"->ToString@limit]
	];
	
	If[KeyExistsQ[args,"Elements"],
		element = Lookup[args,"Elements"];
		If[StringQ[element],element = {element}];
		
		Switch[element,
			{("Images"|"Data"|"FullData"|"ImageLinks"|"LinkedThumbnails")..},
				element,
			_,
				Message[ServiceExecute::nval,"Elements","Flickr"];
				Throw[$Failed]
			]
		,
		element = {"Data"}];

	If[MemberQ[element,("Data"|"FullData")] && KeyExistsQ[args,"Format"],
		format = Lookup[args,"Format"];
		Switch[format,
			("JSON"|"XML"|"Association"|"Dataset"),
				format,
			_,
				Message[ServiceExecute::nval,"Format","Flickr"];
				Throw[$Failed]
			],
		format = "Dataset"
	];

	If[format != "XML",rawparams = Join[rawparams,{"format" -> "json", "nojsoncallback" -> "1"}]];

	rawparams = Join[rawparams,{"extras" -> "description,license,date_upload,date_taken,owner_name,original_format,last_update,geo,tags,machine_tags,views"}];

	If[format=="XML",
		rawdata = FromCharacterCode[OAuthClient`rawoauthdata[id,"RawAlbumImages",rawparams],"UTF-8"],
		rawdata = flickrimport[FromCharacterCode[OAuthClient`rawoauthdata[id,"RawAlbumImages",rawparams],"UTF-8"]];
	];

	photosInfo = rawdata["photoset"]["photo"];
	result = Rule[#,Switch[#,
				"Images", 
				(
					If[KeyExistsQ[args,"ImageSize"],
						size = Lookup[args,"ImageSize"];
						If[!MatchQ[size,"Small" | "Medium" | "Large" | "Thumbnail" | "Original"],
							Message[ServiceExecute::nval,"ImageSize","Flickr"];
							Throw[$Failed]
						];
						,
						size  = "Medium"
					];
					size = mapImageSizes[size];	
					getPhoto[#, size, True]& /@ photosInfo
				),
				"Data",
				(
					Switch[format,
						"JSON",
						photosInfo,
						"XML",
						rawdata,
						"Association",
						<|"Thumbnail" -> getPhoto[#, "t", True], "Keys"-> <|"ID" -> #["id"], "Farm" -> #["farm"], "Server" -> #["server"], "Secret" -> #["secret"]|>,
						"Title" -> #["title"]|> & /@ photosInfo,
						_,
						outputData = <|"Thumbnail" -> getPhoto[#, "t", True], "Keys"-> <|"ID" -> #["id"], "Farm" -> #["farm"], "Server" -> #["server"], "Secret" -> #["secret"]|>,
						"Title" -> #["title"]|> & /@ photosInfo;
						Dataset[outputData]
					]
				),
				"FullData",
				(
					Switch[format,
						"JSON",
						rawdata,
						"XML",
						rawdata,
						"Association",
						outputData = Block[{var = #},
							Association@RotateRight[
								FilterRules[
									KeyMap[Replace[#,{"owner" -> "Owner", "ispublic" -> "IsPublic", "title" -> "Title", "license" -> "License",
										"ownername" -> "Ownername", "lastupdate" -> "LastUpdate", "latitude" -> "Latitude", "longitude" -> "Longitude", "tags" -> "Tags", "machine_tags" -> "MachineTags", "views" -> "Views", "place_id" -> "PlaceID"}]&,
										AssociateTo[var, {"description" -> var["description"]["_content"],
										"OriginalFormat"->ToUpperCase[Lookup[var,"originalformat","JPG"]], "DateTaken"->Quiet[Check[formatDate3[var["datetaken"]], var["datetaken"]]],
										"DateUploaded"->FromUnixTime[ToExpression[var["dateupload"]]],"LastUpdate"->FromUnixTime[ToExpression[var["lastupdate"]]],
										"Keys" -> <|"ID" -> var["id"], "Farm" -> var["farm"], "Server" -> var["server"], "Secret" -> var["secret"]|>,
										"Thumbnail" -> getPhoto[var, "t", True]}
										]
									],
								{"DateTaken", "DateUpload", "Description", "IsPublic", "Keys", "LastUpdate", "Latitude", "License", "Longitude", "MachineTags", "OriginalFormat", "Owner", "Ownername", "PlaceID", "Tags", "Thumbnail", "Title", "Views"}
								]
							]
						] & /@ photosInfo;
						formatImageFullData/@outputData,
						_,
						outputData = Block[{var = #},
							Association@RotateRight[
								FilterRules[
									KeyMap[Replace[#,{"owner" -> "Owner", "ispublic" -> "IsPublic", "title" -> "Title", "license" -> "License",
										"ownername" -> "Ownername", "lastupdate" -> "LastUpdate", "latitude" -> "Latitude", "longitude" -> "Longitude", "tags" -> "Tags", "machine_tags" -> "MachineTags", "views" -> "Views", "place_id" -> "PlaceID"}]&,
										AssociateTo[var, {"description" -> var["description"]["_content"],
										"OriginalFormat"->ToUpperCase[Lookup[var,"originalformat","JPG"]], "DateTaken"->Quiet[Check[formatDate3[var["datetaken"]], var["datetaken"]]],
										"DateUploaded"->FromUnixTime[ToExpression[var["dateupload"]]],"LastUpdate"->FromUnixTime[ToExpression[var["lastupdate"]]],
										"Keys" -> <|"ID" -> var["id"], "Farm" -> var["farm"], "Server" -> var["server"], "Secret" -> var["secret"]|>,
										"Thumbnail" -> getPhoto[var, "t", True]}
										]
									],
								{"DateTaken", "DateUpload", "Description", "IsPublic", "Keys", "LastUpdate", "Latitude", "License", "Longitude", "MachineTags", "OriginalFormat", "Owner", "Ownername", "PlaceID", "Tags", "Thumbnail", "Title", "Views"}
								]
							]
						] & /@ photosInfo;
						Dataset[formatImageFullData/@outputData]
					]
				),
				"ImageLinks",
				(
					If[KeyExistsQ[args,"ImageSize"],
						size = Lookup[args,"ImageSize"];
						If[!MatchQ[size,"Small" | "Medium" | "Large" | "Thumbnail" | "Original"],
							Message[ServiceExecute::nval,"ImageSize","Flickr"];
							Throw[$Failed]
						];
						,
						size  = "Medium"
					];
					size = mapImageSizes[size];
					Hyperlink[getPhoto[#, size, False]]& /@ photosInfo
				),
				"LinkedThumbnails",
				(
					Hyperlink[getPhoto[#, "t", True],getPhotoMainURL[#]]& /@ photosInfo
				),
				_,
				(
					Message[ServiceExecute::nval,"Elements","Flickr"];
					Throw[$Failed]
				)
			]]& /@ element;
	If[Length[result] == 1, result[[1,2]],result]		
]

flickrcookeddata[args___]:=($Failed)

(* Send Message *)
flickrsendmessage[___]:=$Failed

End[] (* End Private Context *)

End[]


SetAttributes[{},{ReadProtected, Protected}];

(* Return three functions to define oauthservicedata, oauthcookeddata, oauthsendmessage  *)
{Flickr`Private`flickrdata,Flickr`Private`flickrcookeddata,Flickr`Private`flickrsendmessage}
