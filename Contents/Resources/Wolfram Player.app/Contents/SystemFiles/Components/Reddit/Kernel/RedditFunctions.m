BeginPackage["RedditFunctions`"]

urlQ::usage = "";
RLangCodes::usage = "";
RFormatUserInformation::usage = "";
RFormatKarma::usage = "";
RFormatTrophy::usage = "";
RFormatFriendsList::usage = "";
RFormatFlair::usage = "";
RFormatComment::usage = "";
RFormatLink::usage = "";
RFormatMessage::usage = "";
RFormatSubreddit::usage = "";
RFormatListing::usage = "";
RFormatFullname::usage = "";
RParseURL::usage = "";
RParsePostFromURL::usage = "";
RParsePostFromURL2::usage = "";
RParseCommentFromURL::usage = "";
ShowThumbnails::usage = "";

Begin["`Private`"] (* Begin Private Context *) 

camelCase[l_List, rest___]:=camelCase[#,rest]&/@l

camelCase[str_String, separators_:{"_"}]:=StringReplace[
 StringReplace[
  StringReplace[str, 
   Thread[separators -> " "]], {WordBoundary ~~ word_ :> 
    ToUpperCase[word]}], {"Id"~~WordBoundary->"ID",WhitespaceCharacter -> "","Css"~~WordBoundary->"CSS",
    "Url"~~WordBoundary->"URL","Urls"~~WordBoundary->"URLs","Html"~~WordBoundary -> "HTML","Img"~~WordBoundary -> "IMG",
    "Utc"~~WordBoundary->"UTC","Sr"~~WordBoundary -> "Subreddit"}]

(*This auxiliar function is used to make the camelization lists*)

(**************************************** URL Parser **********************************************************)

toBase36[name_Integer] := IntegerString[name,36]
toBase36[n___] := n 

urlQ[url_]:= (MatchQ[URLParse[url,"Domain"], "reddit.com" | "www.reddit.com"]) (* Checks for valid reddit url *)

RParseURL[url_String]:=Block[{path,sr,link,comment},
	path = Replace[URLParse[url,"Path"],"":>Nothing,{1}];
	Switch[path,
		 {},
			sr = None; link = None; comment = None,
		{"r",_String},
			sr = Last@path; link = None; comment = None,
		{"r",_String,"comments",_String,_String},
			sr = path[[2]]; link = path[[4]]; comment = None,
		{"r",_String,"comments",_String,_String,_String},
			sr = path[[2]]; link = path[[4]]; comment = path[[6]],
		 _,
			Message[ServiceExecute::rdivurl,url];
			Throw[$Failed]
	];
	{sr,link,comment}
] (* This parses any reddit url to its {subreddit,post,comment} form *)

RParsePostFromURL[url_String]:=Block[{path,sr,link},
	If[urlQ[url],
		path = Replace[URLParse[url,"Path"],"":>Nothing,{1}];
		Switch[path,
			{"r",_String,"comments",_String},
				link = "t3_" <> path[[4]],
			{"r",_String,"comments",_String,_String},
				link = "t3_" <> path[[4]],
			{"r",_String,"comments",_String,_String,_String},
				link = "t3_" <> path[[4]],
			 _,
				Message[ServiceExecute::rdivurl,url];
				Throw[$Failed]
		];
		link,
		url
	]
]	(* This parses any reddit url, validates that it is one from a post and returns its encoded fullname, otherwise it returns the same string *)

RParsePostFromURL2[url_String]:=Block[{path,sr,link},
	If[urlQ[url],
		path = Replace[URLParse[url,"Path"],"":>Nothing,{1}];
		Switch[path,
			{"r",_String,"comments",_String},
				link = path[[4]],
			{"r",_String,"comments",_String,_String},
				link = path[[4]],
			{"r",_String,"comments",_String,_String,_String},
				link = path[[4]],
			 _,
				Message[ServiceExecute::rdivurl,url];
				Throw[$Failed]
		];
		link,
		StringReplace[url,"t3_" ~~ r : __ :> r]
	]
]	(* This parses any reddit url, validates that it is one from a post and returns its non-encoded fullname, otherwise it decodes the fullname *)

RParseCommentFromURL[url_String]:=Block[{path,sr,link,comment},
	If[urlQ[url],
		path = Replace[URLParse[url,"Path"],"":>Nothing,{1}];
		Switch[path,
			{"r",_String,"comments",_String,_String,_String},
				link = path[[4]]; comment = path[[6]],
			 _,
				Message[ServiceExecute::rdivurl,url];
				Throw[$Failed]
		];
		{link,comment},
		StringReplace[url,"t1_" ~~ r : __ :> r]
	]
]	(* This parses any reddit url, validates that it is one from a comment and returns its encoded fullname, otherwise it decodes the fullname *)

(****************************************************************************** user and account format ******************************************************************************)

unixbase = AbsoluteTime[{1970,1,1,0,0,0}];

RFormatUserInformation[data_] := Block[{var = data},
 	KeyDropFrom[var, {"created", "modhash","features"}];
 	KeySortBy[Which[# === "ID", 1, # === "Name", 2, # === "CreationDate", 3, True, 4] &]@Association@KeyValueMap[
 		Which[
 			#1 === "created_utc",
 				"CreationDate" -> If[NumberQ[#2], FromUnixTime[#2, TimeZone -> 0], #2],
 			#1 === "gold_expiration",
 				"GoldExpiration" -> If[NumberQ[#2], FromUnixTime[#2, TimeZone -> 0], Missing["NotApplicable"]],
 			#1 === "suspension_expiration_utc",
 				"SuspensionExpirationDate" -> If[NumberQ[#2], FromUnixTime[#2, TimeZone -> 0], Missing["NotApplicable"]],
 			#2 === Null,
 				(Replace[#1, RCamelUserInformation]) -> Missing["NotAvailable"],
 			True,
 				(Replace[#1, RCamelUserInformation]) -> #2
 		]&,var]
]

RFormatUserInformation["Template",user_String]:= <|"ID" -> Missing["InvalidUsername"], "Name" -> user, "CreationDate" -> Missing["InvalidUsername"],
	"CommentKarma" -> Missing["InvalidUsername"], "HasVerifiedEmail" -> Missing["InvalidUsername"], "HideFromRobots" -> Missing["InvalidUsername"],
	"IsFriend" -> Missing["InvalidUsername"], "IsGold" -> Missing["InvalidUsername"], "IsMod" -> Missing["InvalidUsername"], "LinkKarma" -> Missing["InvalidUsername"]|>

RCamelUserInformation = Dispatch[{"has_mail" -> "HasMail", "name" -> "Name", "is_friend" -> "IsFriend",  "hide_from_robots" -> "HideFromRobots",  "is_suspended" -> "IsSuspended",
	"modhash" -> "Modhash", "has_mod_mail" -> "HasModMail",  "link_karma" -> "LinkKarma", "comment_karma" -> "CommentKarma", "gold_creddits" -> "GoldCreddits",
	"over_18" -> "Over18", "in_beta" -> "InBeta", "is_employee" -> "IsEmployee", "is_gold" -> "IsGold", "is_mod" -> "IsMod", "inbox_count" -> "InboxCount",
	"has_verified_email" -> "HasVerifiedEmail", "id" -> "ID"}]

RFormatKarma[data_Association]:= AssociationThread[Rule[{"Subreddit", "CommentKarma", "LinkKarma"}, Values[data]]]

RFormatKarma[sr_String]:= Association["Subreddit"->sr,"CommentKarma"->0,"LinkKarma"->0]

RFormatKarma[] = Association["Subreddit"->Missing["NonExistent"],"CommentKarma"->0,"LinkKarma"->0]

RFormatTrophy[data_Association]:= Block[{var = data},
	KeyDropFrom[var, {"icon_40","award_id","url"}];
	KeySortBy[Which[# === "ID", 1, # === "Name", 2, # === "Description",3, # === "Icon", 4] &]@Association@KeyValueMap[
		Which[
			#1 === "icon_70",
				"Icon" -> If[#2 === "", Missing["NotAvailable"], #2],
			#2 === Null, 
				(Replace[#1, RCamelTrophy]) -> Missing["NotAvailable"],
			True,
				(Replace[#1, RCamelTrophy]) -> #2
			]&,var]
]

RFormatTrophy[data_Association,rules_List]:= Block[{var = data},
	KeyDropFrom[var, {"icon_40","icon_70","url"}];
	KeySortBy[Which[# === "Icon", 1, # === "ID", 2, # === "Name",3, # === "Description", 4] &]@Association@KeyValueMap[
		Which[
			#1 === "award_id",
				"Icon" -> Replace[#2,rules],
			#2 === Null, 
				(Replace[#1, RCamelTrophy]) -> Missing["NotAvailable"],
			True,
				(Replace[#1, RCamelTrophy]) -> #2
			]&,var]
]

RCamelTrophy = Dispatch[{"description" -> "Description", "url" -> "URL", "award_id" -> "AwardID", "id" -> "ID", "name" -> "Name"}]

RFormatFriendsList[data_Association] := Block[{var = data},
	KeySortBy[Which[# === "ID", 1, # === "Name", 2, # === "Date", 3] &]@Association@KeyValueMap[
		Which[
			#1 === "id",
				"ID" -> StringDrop[#2,3],
			#1 === "date",
				"Date" -> If[NumberQ[#2], DateObject[(unixbase + #2), "TimeZone" -> 0, DateFormat -> "DateShort"], #2],
			#2 === Null,				
				(Replace[#1, RCamelFriendlist]) -> Missing["NotAvailable"],
			True,
				(Replace[#1, RCamelFriendlist]) -> #2
		]&,var]
]

RCamelFriendlist = Dispatch[{"name"->"Name"}]

RFormatFlair[data_] := Block[{},
	Association@ RotateLeft[Replace[Thread[Rule[Keys[#], Map[Which[Last@# === Null, Missing["NotAvailable"], True, Last@#] &, #]]] &@ data,
		{"flair_css_class" -> "FlairCSSClass", "user" -> "User", "flair_text" -> "FlairText",
		"flair_template_id" -> "FlairTemplateID", "flair_text_editable" -> "FlairTextEditable",
		"flair_position" -> "FlairPosition"}, {2}]]]

(****************************************************************************** templates ******************************************************************************)

RSubredditTemplate[id_]:= <|"BannerIMG" -> Missing["InvalidSubreddit"], "UserIsBanned" -> Missing["InvalidSubreddit"], "wiki_enabled" -> Missing["InvalidSubreddit"], 
 "ID" -> Missing["InvalidSubreddit"], "UserIsContributor" -> Missing["InvalidSubreddit"], "DisplayName" -> Missing["InvalidSubreddit"], "HeaderIMG" -> Missing["InvalidSubreddit"], 
 "Title" -> Missing["InvalidSubreddit"], "CollapseDeletedComments" -> Missing["InvalidSubreddit"], "PublicDescription" -> Missing["InvalidSubreddit"], "Kind" -> Missing["InvalidSubreddit"],
 "Over18" -> Missing["InvalidSubreddit"], "IconSize" -> Missing["InvalidSubreddit"], "SuggestedCommentSort" -> Missing["InvalidSubreddit"], "IconIMG" -> Missing["InvalidSubreddit"], 
 "HeaderTitle" -> Missing["InvalidSubreddit"], "Description" -> Missing["InvalidSubreddit"], "UserIsMuted" -> Missing["InvalidSubreddit"], "SubmitLinkLabel" -> Missing["InvalidSubreddit"], 
 "AccountsActive" -> Missing["InvalidSubreddit"], "PublicTraffic" -> Missing["InvalidSubreddit"], "HeaderSize" -> Missing["InvalidSubreddit"], "Subscribers" -> Missing["InvalidSubreddit"], 
 "SubmitTextLabel" -> Missing["InvalidSubreddit"], "Lang" -> Missing["InvalidSubreddit"], "KeyColor" -> Missing["InvalidSubreddit"], "GlobalID" -> Missing["InvalidSubreddit"], 
 "URL" -> Missing["InvalidSubreddit"], "Quarantine" -> Missing["InvalidSubreddit"], "HideAds" -> Missing["InvalidSubreddit"], "CreationDate" -> Missing["InvalidSubreddit"], 
 "BannerSize" -> Missing["InvalidSubreddit"], "UserIsModerator" -> Missing["InvalidSubreddit"], "UserSubredditThemeEnabled" -> Missing["InvalidSubreddit"], "CommentScoreHideMins" -> Missing["InvalidSubreddit"], 
 "SubredditType" -> Missing["InvalidSubreddit"], "SubmissionType" -> Missing["InvalidSubreddit"], "UserIsSubscriber" -> Missing["InvalidSubreddit"]|>

(****************************************************************************** fullnames formatting ******************************************************************************)

(*

	KeyValueMap is used to modify the "Null" values to Missing["error"] with customized
	errros based on the Keys, this allows for a more explanatory data output which could
	be useful for the users of the connection. It also helps to camelize the Keys.

	The keys that end in '_html' return the html source of the reddit object, importing it
	provides a way to extract it and display or export it on an html viewer if desired.
	Perhaps those keys could be removed, but, for now I've left them for reference
	
*)

RFormatComment[data_]:= Block[{var=data},
	KeyDropFrom[var,{"created","body_html"}];
	AssociateTo[var,"Kind"->"t1"];
	Association@KeyValueMap[
		Which[
			#1 === "created_utc",
				"CreationDate" -> If[NumberQ[#2], FromUnixTime[#2, TimeZone -> 0], #2],
			#1 === "replies" && MatchQ[#2,""],
				"Replies" -> {},
			#1 === "replies" && ListQ[#2],
				"Replies" -> Flatten[#2],
			#2===Null && MemberQ[{"removal_reason","banned_by","approved_by"},#1],
				(Replace[#1, RCamelComment]) -> Missing["NotApplicable"],
			#2 === Null,
				(Replace[#1, RCamelComment]) -> Missing["NotAvailable"],
			True,
				(Replace[#1, RCamelComment]) -> #2
		]&,var]
]

RCamelComment = Dispatch[{"subreddit_id" -> "SubredditID", "link_title" -> "LinkTitle", "banned_by" -> "BannedBy", "removal_reason" -> "RemovalReason", "link_id" -> "LinkID",
	"link_author" -> "LinkAuthor", "likes" -> "Likes", "replies" -> "Replies", "user_reports" -> "UserReports", "saved" -> "Saved", "id" -> "ID", "gilded" -> "Gilded",
	"archived" -> "Archived", "stickied" -> "Stickied", "author" -> "Author", "parent_id" -> "ParentID", "score" -> "Score", "approved_by" -> "ApprovedBy", "over_18" -> "Over18", 
	"report_reasons" -> "ReportReasons", "controversiality" -> "Controversiality", "body" -> "Body", "edited" -> "Edited", "author_flair_css_class" -> "AuthorFlairCSSClass", 
	"downs" -> "Downs", "quarantine" -> "Quarantine", "subreddit" -> "Subreddit", "score_hidden" -> "ScoreHidden", "name" -> "GlobalID", "author_flair_text" -> "AuthorFlairText", 
	"link_url" -> "LinkURL", "ups" -> "Ups", "mod_reports" -> "ModReports", "num_reports" -> "ReportsCount", "distinguished" -> "Distinguished", "depth" -> "Depth",
	"subreddit_name_prefixed" -> "SubredditNamePrefix"}]

RFormatLink[data_]:= Block[{var=data},
	KeyDropFrom[var,{"created","selftext_html"}];
	AssociateTo[var,"Kind"->"t3"];
	If[!KeyExistsQ[var,"preview"],AssociateTo[var,"preview"->None]];
	KeySortBy[Which[
		# === "ID", 1, # === "GlobalID", 1.25, MatchQ[#, "Kind"|"Title"], 1.5, # === "URL", 2, # === "Permalink", 2.5,
		# === "Author", 1.75, # === "CreationDate", 2.25, MatchQ[#, "Subreddit"|"SubredditGlobalID"], 2.5, MatchQ[#, "Thumbnail"|"Preview"], 2.75,
		# === "Ups", 3, # === "Downs", 3.25, # === "Score", 3.5, # === "CommentsCount", 3.75, MatchQ[#, "Saved"|"Edited"], 4, True, 5] &]@
	Association@KeyValueMap[
		Which[
			#1 === "created_utc",
				"CreationDate" -> If[NumberQ[#2], FromUnixTime[#2, TimeZone -> 0], #2],
			#1 === "thumbnail",
				If[RedditFunctions`ShowThumbnails,
					"Thumbnail" -> (Block[{img = Quiet[Import[#]]}, If[FailureQ[img], Missing["NotAvailable"], img]]& @ #2),
					"Thumbnail"-> If[#2 === "", Missing["NotAvailable"], #2]],
 			#1 === "permalink",
 				"Permalink" -> "http://www.reddit.com"<>#2,
  			#1 === "preview",
 				"Preview" -> If[#2 === None, Missing["NotApplicable"], <|"Images"-> formatimage /@ #2["images"]|>],
			#2===Null && MemberQ[{"removal_reason","banned_by","approved_by"},#1],
				(Replace[#1, RCamelLink]) -> Missing["NotApplicable"],
			#2 === Null,
				(Replace[#1, RCamelLink]) -> Missing["NotAvailable"],
			True,
				(Replace[#1, RCamelLink]) -> #2
		]&,var]
]

formatimage[img_Association] := <|"ID" -> img["id"], 
  "Source" -> formatimage0[img["source"]], 
  "Resolutions" -> (formatimage0 /@ img["resolutions"]), 
  "Variants" -> img["variants"]|>

formatimage0[img_Association] := <|
  "URL" -> ImportString[img["url"], "HTML"], "Width" -> img["width"], 
  "Height" -> img["height"]|>

RCamelLink = Dispatch[{"created_utc" -> "CreatedUTC", "domain" -> "Domain", "banned_by" -> "BannedBy", "media_embed" -> "MediaEmbed", "upvote_ratio" -> "UpvoteRatio",
	"subreddit" -> "Subreddit", "selftext" -> "Selftext", "likes" -> "Likes", "suggested_sort" -> "SuggestedSort", "user_reports" -> "UserReports",
	"secure_media" -> "SecureMedia", "link_flair_text" -> "LinkFlairText", "id" -> "ID", "from_kind" -> "FromKind", "gilded" -> "Gilded", "archived" -> "Archived",
	"clicked" -> "Clicked", "report_reasons" -> "ReportReasons", "author" -> "Author","media" -> "Media", "score" -> "Score","approved_by" -> "ApprovedBy", "over_18" -> "Over18",
	"hidden" -> "Hidden", "num_comments" -> "CommentsCount", "thumbnail" -> "Thumbnail",	"subreddit_id" -> "SubredditGlobalID","hide_score" -> "HideScore", "edited" -> "Edited",
	"link_flair_css_class" -> "LinkFlairCSSClass", "author_flair_css_class" -> "AuthorFlairCSSClass", "downs" -> "Downs","secure_media_embed" -> "SecureMediaEmbed", "saved" -> "Saved",
	"removal_reason" -> "RemovalReason", "stickied" -> "Stickied", "from" -> "From", "is_self" -> "IsSelf",	"from_id" -> "FromID", "locked" -> "Locked", "name" -> "GlobalID",
	"url" -> "URL", "author_flair_text" -> "AuthorFlairText", "quarantine" -> "Quarantine",	"title" -> "Title", "distinguished" -> "Distinguished",	"mod_reports" -> "ModReports",
	"visited" -> "Visited", "num_reports" -> "ReportsCount", "ups" -> "Ups", "preview" -> "Preview", "post_hint" -> "PostHint", "brand_safe" -> "BrandSafe",
	"contest_mode" -> "ContestMode", "spoiler" -> "Spoiler", "subreddit_name_prefixed" -> "SubredditNamePrefix", "subreddit_type" -> "SubredditType"}]

RFormatMessage[data_]:= Block[{var=data},
	KeyDropFrom[var,{"created","body_html","replies"}];
	AssociateTo[var,"Kind"->"t4"];
	Association@KeyValueMap[
		Which[
 			#1 === "created_utc",
 				"CreationDate" -> If[NumberQ[#2], FromUnixTime[#2, TimeZone -> 0], #2],
			#2 === Null,
				(Replace[#1, RCamelMessage]) -> Missing["NotAvailable"],
			#1 === "first_message",
				"FirtMessageID" -> toBase36[#2],
			True,
				(Replace[#1, RCamelMessage]) -> #2
		]&,var]
]

RCamelMessage = Dispatch[{"body" -> "Body", "was_comment" -> "IsPostReply", "first_message" -> "FirstMessage", "name" -> "GlobalID", "first_message_name" -> "FirstMessageGlobalID",
	"dest" -> "Recipient", "author" -> "Sender", "subreddit" -> "Subreddit", "likes" -> "Likes", "parent_id" -> "ParentID", "context" -> "Context", "id" -> "ID", "new" -> "New",
	"distinguished" -> "Distinguished", "subject" -> "Subject", "link_title" -> "PostTitle"}]

RFormatSubreddit[data_]:= Block[{var=data},
	AssociateTo[var,"lang"->Replace[#,RLangCodes]&@var["lang"]];
	AssociateTo[var,"Kind"->"t5"];
	KeyDropFrom[var, {"created","submit_text","submit_text_html","description_html","public_description_html"}];
	KeySortBy[Which[
		# === "ID", 1, # === "GlobalID", 1.25, MatchQ[#, "Kind"|"Title"], 1.5, # === "URL", 1.75, # === "CreationDate", 2, MatchQ[#, "DisplayName"|"Language"], 2.25,
		# === "Subscribers", 2.5, # === "SubredditType", 2.75, MatchQ[#, "SubmissionType"|"SubmitLinkLabel"|"SubmitTextLabel"], 3,
		# === "IconImage", 3.25, # === "PublicDescription", 3.5, # === "Description", 3.75, # === "Over18", 4, True, 5] &]@
	Association@KeyValueMap[
		Which[
 			#1 === "created_utc",
 				"CreationDate" -> If[NumberQ[#2], FromUnixTime[#2, TimeZone -> 0], #2],
 			#1 === "url",
 				"URL" -> "http://www.reddit.com"<>#2,
 			MemberQ[{"header_img","banner_img","icon_img"},#1],
 				If[RedditFunctions`ShowThumbnails,(Replace[#1, RCamelSubreddit]) -> If[#2 === "", Missing["NotAvailable"], Import[#2]],Replace[#1, RCamelSubreddit]->If[#2 === "", Missing["NotAvailable"], #2]],
			#2 === Null,
				(Replace[#1, RCamelSubreddit]) -> Missing["NotAvailable"],
			True,
				(Replace[#1, RCamelSubreddit]) -> #2
		]&,var]
]

RCamelSubreddit = Dispatch[{"banner_img" -> "BannerImage", "user_is_banned" -> "UserIsBanned", "id" -> "ID", "user_is_contributor" -> "UserIsContributor", "display_name" -> "DisplayName",
	"header_img" -> "HeaderImage", "title" -> "Title", "collapse_deleted_comments" -> "CollapseDeletedComments", "public_description" -> "PublicDescription", "over18" -> "Over18",
	"community_rules" -> "CommunityRules", "icon_size" -> "IconSize", "suggested_comment_sort" -> "SuggestedCommentSort", "icon_img" -> "IconImage", "header_title" -> "HeaderTitle",
	"description" -> "Description", "user_is_muted" -> "UserIsMuted", "submit_link_label" -> "SubmitLinkLabel", "accounts_active" -> "AccountsActive", "public_traffic" -> "PublicTraffic",
	"header_size" -> "HeaderSize", "subscribers" -> "Subscribers", "submit_text_label" -> "SubmitTextLabel", "lang" -> "Language", "key_color" -> "KeyColor", "name" -> "GlobalID",
	"url" -> "URL", "quarantine" -> "Quarantine", "hide_ads" -> "HideAds", "created_utc" -> "CreatedUTC", "banner_size" -> "BannerSize", "user_is_moderator" -> "UserIsModerator",
	"user_sr_theme_enabled" -> "UserSubredditThemeEnabled", "comment_score_hide_mins" -> "CommentScoreHideMins", "subreddit_type" -> "SubredditType", "submission_type" -> "SubmissionType",
	"user_is_subscriber" -> "UserIsSubscriber", "wiki_enabled" -> "WikiEnabled"}]

RFormatListing[data_] := Lookup[data, "children"]

RFormatFullname[data_]:= Switch[data["kind"],
							"t1",
								RFormatComment[data["data"]],
							"t3",
								RFormatLink[data["data"]],
							"t4",
								RFormatMessage[data["data"]],
							"t5",
								RFormatSubreddit[data["data"]],
							"listing" | "Listing" | "more",
								RFormatListing[data["data"]]
]

(****************************************************************************** code to language entity ******************************************************************************)

RLangCodes = Dispatch[{"en" -> Entity["Language", "English"], "ar" -> Entity["Language", "Arabic"], "be" -> Entity["Language", "Belarusan"], "bg" -> Entity["Language", "Bulgarian"], "bs" -> Entity["Language", "Bosnian"], 
 "ca" -> Entity["Language", "CatalanValencianBalear"], "cs" -> Entity["Language", "Czech"], "da" -> Entity["Language", "Danish"], "de" -> Entity["Language", "German"], "el" -> Entity["Language", "Greek"], 
 "en-au" -> Entity["Language", "English"], "en-ca" -> Entity["Language", "English"], "en-gb" -> Entity["Language", "English"], "en-us" -> Entity["Language", "English"], "eo" -> Entity["Language", "Esperanto"], 
 "es" -> Entity["Language", "Spanish"], "es-ar" -> Entity["Language", "Spanish"], "et" -> Entity["Language", "Estonian"], "eu" -> Entity["Language", "Basque"], "fa" -> Entity["Language", "FarsiEastern"], 
 "fi" -> Entity["Language", "Finnish"], "fr" -> Entity["Language", "French"], "gd" -> Entity["Language", "ScottishGaelic"], "he" -> Entity["Language", "Hebrew"], "hi" -> Entity["Language", "Hindi"], 
 "hr" -> Entity["Language", "Croatian"], "hu" -> Entity["Language", "Hungarian"], "hy" -> Entity["Language", "Armenian"], "id" -> Entity["Language", "Indonesian"], "is" -> Entity["Language", "Icelandic"], 
 "it" -> Entity["Language", "Italian"], "ja" -> Entity["Language", "Japanese"], "kn_IN" -> Entity["Language", "Kannada"], "ko" -> Entity["Language", "Korean"], "la" -> Entity["Language", "Latin"], "leet" -> "leet", 
 "lol" -> "lol", "lt" -> Entity["Language", "Lithuanian"], "lv" -> Entity["Language", "Latvian"], "nl" -> Entity["Language", "Dutch"], "nn" -> Entity["Language", "NorwegianNynorsk"], "no" -> Entity["Language", "Norwegian"], 
 "pir" -> "pir", "pl" -> Entity["Language", "Polish"], "pt" -> Entity["Language", "Portuguese"], "pt-pt" -> Entity["Language", "Portuguese"], "pt_BR" -> Entity["Language", "Portuguese"], 
 "ro" -> Entity["Language", "Romanian"], "ru" -> Entity["Language", "Russian"], "sk" -> Entity["Language", "Slovak"], "sl" -> Entity["Language", "Slovenian"], "sr" -> Entity["Language", "Serbian"], 
 "sr-la" -> Entity["Language", "Serbian"], "sv" -> Entity["Language", "Swedish"], "ta" -> Entity["Language", "Tamil"], "th" -> Entity["Language", "Thai"], "tr" -> Entity["Language", "Turkish"], 
 "uk" -> Entity["Language", "Ukrainian"], "vi" -> Entity["Language", "Vietnamese"], "zh" -> Entity["Language", "Chinese"]}]


End[] (* End Private Context *)

EndPackage[]