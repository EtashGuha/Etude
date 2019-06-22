BeginPackage["FacebookFunctions`"]

camelCase::usage = "";
FFormatUserInformation::usage = "";
FFormatBMM::usage = "";
FFormatPageInformation::usage = "";
FFormatTaggedPost::usage = "";
FFormatFeedInformation::usage = "";
FFormatPost::usage = "";

Begin["`Private`"] (* Begin Private Context *) 

camelCase[l_List, rest___]:=camelCase[#,rest]&/@l

camelCase[str_String, separators_:{"_"}]:=StringReplace[
 StringReplace[
  StringReplace[str, 
   Thread[separators -> " "]], {WordBoundary ~~ word_ :> 
    ToUpperCase[word]}], {"Id"~~WordBoundary->"ID",WhitespaceCharacter -> "",
    "Url"~~WordBoundary->"URL","Urls"~~WordBoundary->"URLs"}]

(*This auxiliary function is used to make the camelization lists*)

(****************************************************************************** user and account format ******************************************************************************)

unixbase = AbsoluteTime[{1970,1,1,0,0,0}];

FFormatUserInformation[user_] := (
 	Association@KeyValueMap[
		Which[
			#1 === "birthday",
				"Birthday" -> DateObject[#2],
			#1 === "cover",
				"CoverLink" -> formatCover@#2,
			#1 === "currency",
				"Currency" -> formatCurrency@#2,
			#1 === "devices",
				"Devices" -> formatDevice/@#2,
			#1 === "education",
				"Education" -> formatEducationExperience/@#2,
			#1 === "interested_in",
				"InterestedIn" -> Capitalize/@#2,
			MatchQ[#1,"favorite_athletes"|"favorite_teams"|"inspirational_people"|"languages"|"sports"],
				Replace[#1, {"favorite_athletes"->"FavoriteAthletes", "favorite_teams"->"FavoriteTeams",
					"inspirational_people"-> "InspirationalPeople", "languages"->"Languages",
					"sports"->"Sports"}] -> formatSimple/@#2,
			MatchQ[#1,"location"|"hometown"|"significant_other"],
				Replace[#1, {"location"->"Location", "hometown"->"Hometown","significant_other"->"SignificantOther"}] -> formatSimple@#2,
			#1 === "gender",
				"Gender" -> Capitalize[#2],
			#1 === "picture",
				"PictureLink" -> URL[#2["data"]["url"]],
			MatchQ[#1,"link"|"website"],
				Replace[#1, {"link"->"Link", "website"->"Website"}] -> URL[#2],
			#1 === "updated_time",
				"UpdatedTime" -> DateObject[{StringInsert[#2,":",-3], {"ISODateTime", "ISOTimeZone"}}],
			#1 === "work",
				"Work" -> formatWorkExperience/@#2,
			True,
				(Replace[#1, FCamelUserInformation]) -> #2
		]&, user]
)

FCamelUserInformation = Dispatch[{"about"->"About", "email"->"Email", "first_name"->"FirstName", "id"->"UserID", "installed"->"WolframConnectedQ",
 "is_verified"->"IsVerified", "last_name"->"LastName", "locale"->"Locale", "middle_name"->"MiddleName", "name"->"FullName", "picture"->"Picture",
 "quotes"->"Quotes", "relationship_status"->"RelationshipStatus", "third_party_id"->"ThirdPartyID", "timezone"->"Timezone", "verified"->"Verified"}]

formatUserSimple[s_]:= (KeyMap[Replace[{"id"->"UserID", "name"->"Name"}]]@KeyTake[{"id", "name"}]@s)
formatSimple[s_]:= (KeyMap[Replace[{"id"->"ID", "name"->"Name"}]]@KeyTake[{"id", "name"}]@s)
formatCover[c_]:= URL[c["source"]]
(*
formatCover[c_]:= (MapAt[URL,{Key["Source"]}]@KeyMap[Replace[{"id"->"ID", "offset_x"->"HorizontalOffset", "offset_y"->"VerticalOffset", "source"->"Source"}]]@KeyDrop[{"cover_id"}]@c)
*)
formatCurrency[c_]:= (KeyMap[Replace[{"currency_offset"->"CurrencyOffset", "usd_exchange"->"USDExchange", "usd_exchange_inverse"->"USDExchangeInverse", "user_currency"->"UserCurrency"}]]@c)
formatDevice[d_]:= (KeyMap[Replace[{"os"->"OS", "hardware"->"Hardware"}]]@d)

formatExperience[e_]:= (
 	Association@KeyValueMap[
		Which[
			#1 === "from",
				"From" -> formatUserSimple@#2,
			#1 === "with",
				"With" -> formatUserSimple/@#2,
			True,
				(Replace[#1, {"id"->"ID","description"->"Description","name"->"Name"}]) -> #2
		]&, e]
)

formatEducationExperience[e_]:= (
 	Association@KeyValueMap[
		Which[
			#1 === "classes",
				"Classes" -> formatExperience/@#2,
			MatchQ[#1, "concentration"|"with"],
				Replace[#1, {"concentration"->"Concentration","with"->"With"}] -> formatSimple/@#2,
			MatchQ[#1, "degree"|"school"|"year"],
				Replace[#1, {"degree"->"Degree","school"->"School","year"->"Year"}] -> formatSimple@#2,
			True,
				(Replace[#1, {"id"->"ID","type"->"Type"}]) -> #2
		]&, e]
)

formatWorkExperience[e_]:= (
 	Association@KeyValueMap[
		Which[
			#1 === "from",
				"From" -> formatUserSimple@#2,
			MatchQ[#1, "start_date"|"end_date"],
				Replace[#1, {"start_date"->"StartDate","end_date"->"EndDate"}] -> DateObject[#2],
			MatchQ[#1, "employer"|"location"|"position"],
				Replace[#1, {"employer"->"Employer","location"->"Location","position"->"Position"}] -> formatSimple@#2,
			#1 === "projects",
				"Projects" -> formatExperience/@#2,
			#1 === "with",
				"With" -> formatUserSimple/@#2,
			True,
				(Replace[#1, {"id"->"ID","description"->"Description"}]) -> #2
		]&, e]
)

formatProjectExperience[e_]:= (
 	Association@KeyValueMap[
		Which[
			#1 === "from",
				"From" -> formatUserSimple@#2,
			MatchQ[#1, "start_date"|"end_date"],
				Replace[#1, {"start_date"->"StartDate","end_date"->"EndDate"}] -> DateObject[#2],
			#1 === "with",
				"With" -> formatUserSimple/@#2,
			True,
				(Replace[#1, {"id"->"ID","description"->"Description","name"->"Name"}]) -> #2
		]&, e]
)

FFormatBMM[prop_,data_]:= (
 	Association@KeyValueMap[
		Which[
			#1 === "id",
				prop <> "ID" -> #2,
			MatchQ[#1, "updated_time"|"created_time"],
				Replace[#1, {"updated_time"->"UpdatedTime","created_time"->"CreatedTime"}] -> DateObject[{StringInsert[#2,":",-3], {"ISODateTime", "ISOTimeZone"}}],
			True,
				(Replace[#1, {"name"->"Name"}]) -> #2
		]&, KeyTake[{"id", "name", "updated_time", "created_time"}]@data]
)

FFormatPageInformation[page_] := (
 	Association@KeyValueMap[
		Which[
			#1 === "best_page",
				"BestPage" -> formatSimple@#2,
			#1 === "cover",
				"CoverLink" -> formatCover@#2,
			#1 === "likes",
				"Likes" -> formatSimple/@#2["data"],
			#1 === "location",
				"Location" -> formatLocation@#2,
			#1 === "parking",
				"Parking" -> formatAvailability@#2,
			#1 === "price_range",
				"PriceRange" -> formatPriceRange@#2,
			#1 === "restaurant_services",
				"RestaurantServices" -> formatAvailability@#2,
			#1 === "restaurant_specialties",
				"RestaurantSpecialties" -> formatAvailability@#2,
			MatchQ[#1,"category_list"],
				Replace[#1, {"category_list"->"CategoryList"}] -> formatSimple/@#2,
			MatchQ[#1,"link"|"website"],
				Replace[#1, {"link"->"Link", "website"->"Website"}] -> URL[#2],
			True,
				(Replace[#1, FCamelPageInformation]) -> #2
		]&, page]
)

FCamelPageInformation = Dispatch[{"id"->"PageID", "about"->"About", "attire"->"Attire", "band_members"->"BandMembers", "birthday"->"Birthday",
	"booking_agent"->"BookingAgent", "can_post"->"CanPost", "category"->"Category",	"checkins"->"Checkins",	"company_overview"->"CompanyOverview",
	"current_location"->"CurrentLocation", "description"->"Description", "directed_by"->"DirectedBy", "founded"->"Founded", "general_info"->"GeneralInfo",
	"general_manager"->"GeneralManager", "hometown"->"Hometown", "is_published"->"IsPublished", "is_unclaimed"->"IsUnclaimed", "mission"->"Mission",
	"name"->"PageName", "phone"->"Phone", "press_contact"->"PressContact", "products"->"Products", "talking_about_count"->"TalkingAboutCount",
	"username"->"Username", "were_here_count"->"WereHereCount"}]

formatLocation[loc_]:= With[{long = loc["longitude"]},
 	Association@KeyValueMap[
		Which[
			#1 === "latitude",
				"GeoLocation" -> GeoPosition[{#2, long}],
			True,
				(Replace[#1, FCamelLocationInformation]) -> #2
		]&, KeyDrop[{"longitude"}]@loc]
]

FCamelLocationInformation = Dispatch[{"city"->"City", "city_id"->"CityID", "country"->"Country", "country_code"->"CountryCode", "located_id"->"LocatedID",
	"name"->"Name", "region"->"Region", "region_id"->"RegionID", "state"->"State", "street"->"Street", "zip"->"ZIPCode"}]

formatPlace[place_]:= (MapAt[formatLocation,{Key["Location"]}]@KeyMap[Replace[{"id"->"ID","name"->"Name","location"->"Location"}]]@KeyTake[{"id","name","location"}]@place)

formatPriceRange[pr_]:= Quantity[#, "Dollars"]& /@ ToExpression[StringCases[pr, DigitCharacter ..]]/;StringMatchQ[pr, "$*"]
formatAvailability[rs_]:=(Keys@Select[KeyMap[Capitalize]@rs, MatchQ[True|"1"|1]])

FFormatTaggedPost[tp_]:=(MapAt[DateObject[{StringInsert[#,":",-3], {"ISODateTime", "ISOTimeZone"}}]&,{Key["CreatedTime"]}]@
							KeyMap[Replace[{"id" -> "ID", "message" -> "Message", "created_time" -> "CreatedTime"}]]@
								KeyTake[{"id", "message", "created_time"}]@tp)

FFormatFeedInformation[prop_,data_] := (
 	Association@KeyValueMap[
		Which[
			#1 === "id",
				prop <> "ID" -> #2,
			#1 === "application",
				"Application" -> formatAppSimple@#2,
			#1 === "actions",
				"Actions" -> formatFeedAction/@#2,
			#1 === "comments",
				"Comments" -> formatComment/@#2["data"],
			MatchQ[#1, "updated_time"|"created_time"],
				Replace[#1, {"updated_time"->"UpdatedTime","created_time"->"CreatedTime"}] -> DateObject[{StringInsert[#2,":",-3], {"ISODateTime", "ISOTimeZone"}}],
			MatchQ[#1,"from"|"to"],
				Replace[#1, {"from"->"From","to"->"To"}] -> formatSimple@#2,
			#1 === "likes",
				"Likes" -> formatSimple/@#2["data"],
			#1 === "message_tags",
				"MessageTags" -> formatMessageTag/@#2,
			MatchQ[#1,"icon"|"link"|"picture"|"source"],
				Replace[#1, {"icon"->"Icon","link"->"Link","picture"->"Picture","source"->"Source"}] -> URL@#2,
			#1 === "place",
				"Place" -> formatPlace@#2,
			#1 === "privacy",
				"Privacy" -> formatPrivacy@#2,
			#1 === "with_tags",
				"WithTags" -> formatUserSimple/@#2["data"],
			True,
				(Replace[#1, FCamelFeedInformation]) -> #2
		]&, data]
)

FCamelFeedInformation = Dispatch[{"message"->"Message", "name"->"Name", "caption"->"Caption", "description"->"Description", "type"->"Type",
	"story"->"Story", "comments"->"Comments", "object_id"->"ObjectID", "status_type"->"StatusType"}]

formatAppSimple[app_]:= (MapAt[URL,{Key["Link"]}]@KeyMap[Replace[{"id"->"ID","name"->"Name","link"->"Link"}]]@KeyTake[{"id", "name","link"}]@app)
formatMessageTag[mt_]:= (KeyMap[Replace[{"id"->"ID","name"->"Name","type"->"Type"}]]@KeyTake[{"id","name","type"}]@mt)
formatPrivacy[p_]:= (MapAt[StringSplit[#, ","] &, {{Key["Allow"]}, {Key["Deny"]}}]@ KeyMap[Capitalize]@p)
formatFeedAction[fa_]:= (MapAt[URL,{Key["Link"]}]@KeyMap[Capitalize]@fa)

formatComment[c_]:= (
 	Association@KeyValueMap[
		Which[
			#1 === "from",
				"From" -> formatUserSimple@#2,
			#1 === "created_time",
				"CreatedTime" -> DateObject[{StringInsert[#2,":",-3], {"ISODateTime", "ISOTimeZone"}}],
			True,
				(Replace[#1, {"id"->"CommentID","message"->"Message"}]) -> #2
		]&, KeyTake[{"id","from","message","created_time"}]@c]
)

FFormatPost[data_]:= (
 	Association@KeyValueMap[
		Which[
			#1 === "id",
				"PostID" -> #2,
			MatchQ[#1, "updated_time"|"created_time"],
				Replace[#1, {"updated_time"->"UpdatedTime","created_time"->"CreatedTime"}] -> DateObject[{StringInsert[#2,":",-3], {"ISODateTime", "ISOTimeZone"}}],
			#1 === "picture",
				"PictureLink" -> URL[#2],
			#1 === "comments",
				"Comments" -> formatComment/@#2["data"],
			#1 === "likes",
				"Likes" -> formatPostLike@#2["summary"],
			True,
				(Replace[#1, {"message"->"Message","type"->"Type","story"->"Story"}]) -> #2
		]&, data]
)

formatPostLike[pl_]:=(KeyMap[Replace[{"total_count"->"TotalCount","can_like"->"CanLike","has_liked"->"HasLiked"}]]@pl)


End[] (* End Private Context *)

EndPackage[]