Get["TwitterFunctions.m"]

Begin["TwitterOAuth`"]

ServiceExecute::maximg = "Maximum number of images is 4.";
ServiceExecute::noresp = "The service returned an empty response";
ServiceExecute::rlim = "The request limit for this resource has been reached for the current rate limit window. Please try again in a few minutes.";
ServiceExecute::errimp = "An error has occured when importing data from the service.";
ServiceExecute::unauth = "Not authorized.";
ServiceExecute::notfnd = "Not found.";
ServiceExecute::nouser = "No user matches for specified terms.";

Begin["`Private`"]

(******************************* Twitter *************************************)

(* Authentication information *)

twitterdata[]:=If[ TrueQ[OAuthClient`Private`$UseChannelFramework],
    {
        "OAuthVersion"            -> "1.0a",
        "ServiceName"             -> "Twitter",
        "RequestEndpoint"         -> "https://api.twitter.com/oauth/request_token",
        "AccessEndpoint"          -> "https://api.twitter.com/oauth/access_token",
        "AuthorizeEndpoint"       -> "https://api.twitter.com/oauth/authorize",
        "RedirectURI"             -> "WolframConnectorChannelListen",
        "Blocking"                -> False,
        "VerifierParsing"         -> "oauth_verifier",
        "ClientInfo"              -> {"Wolfram","Token"},
        "AuthenticationDialog"    -> "WolframConnectorChannel",
        "RequestFormat"           -> (Block[{params=Lookup[{##2},"Parameters",{}],method=Lookup[{##2},"Method","GET"]},
                                    URLFetch[#1, {"StatusCode", "Content"}, "Method" -> method, "Parameters" -> params, "Body" -> Lookup[{##2},"BodyData",{}], "MultipartElements" -> Lookup[{##2},"MultipartData",{}], "VerifyPeer" -> True, "CredentialsProvider" -> None]
                                    ]&),
        "Gets"                    -> {"GetTweet","LastTweet","FollowerIDs","FriendIDs","UserData","UserMentions","UserReplies","FollowerMentionNetwork",
            "FriendMentionNetwork","FollowerReplyToNetwork","FriendReplyToNetwork","FriendNetwork","FollowerNetwork","Friends","Followers",
            "RateLimit","UserIDSearch","SearchNetwork","SearchReplyToNetwork","SearchMentionNetwork","UserHashtags","TweetSearch",
            "TweetEventSeries","TweetTimeline","TweetList"},
        "Posts"                   -> {"Tweet","Retweet"},
        "RawGets"                 -> {"RawMentionsTimeline","RawUserTimeline","RawHomeTimeline","RawRetweetTimeline",
                    "RawStatus","RawRetweets","RawRetweeterIDs","RawTweetSearch",
                    "RawDirectMessages","RawDirectMessagesSent","RawDirectMessage",
                    "RawNoRetweetUserIDs", "RawFriendIDs", "RawFollowerIDs",
                    "RawMyFriendship", "RawIncomingFriendships",
                    "RawOutgoingFriendships", "RawFriendship", "RawFriends",
                    "RawFollowers","RawUserSettings","RawVerifyCredentials","RawBlockList","RawBlockIDs",
                    "RawUsers","RawUser","RawUserSearch","RawContributees","RawContributors",
                    "RawSuggestedUsers","RawSuggestedUserCategories","RawSuggestedUserStatuses","RawFavorites","RawAccountStatus"},
        "RawPosts"                -> {
                    "RawDeleteTweet",
                    "RawUpdate","RawRetweet","RawMediaUpload","RawUpload",
                    "RawDeleteDirectMessage","RawSendDirectMessage",
                    "RawUpdateFollowing", "RawStopFollowing", "RawStartFollowing",
                    "RawSetUserSettings","RawUpdateDevice","RawUpdateProfile","RawUpdateBackgroundImage",
                    "RawUpdateProfileColors",
                    (* "RawUpdateProfileImage",*)"RawCreateBlock","RawRemoveBlock","RawAddFavorite","RawRemoveFavorite"},
        "LogoutURL"               -> "https://twitter.com/logout",
        "Information"             -> "A service for sending and receiving tweets from a Twitter account"
    }
    ,
    {
        "OAuthVersion"            -> "1.0a",
        "ServiceName"             -> "Twitter",
        "RequestEndpoint"         -> "https://api.twitter.com/oauth/request_token",
        "AccessEndpoint"          -> "https://api.twitter.com/oauth/access_token",
        "AuthorizeEndpoint"       -> "https://api.twitter.com/oauth/authorize",
        "ClientInfo"              -> {"Wolfram","Token"},
        "AuthenticationDialog"    :> (OAuthClient`tokenOAuthDialog[#, "Twitter", bird]&),
        "RequestFormat"           -> (Block[{params=Lookup[{##2},"Parameters",{}],method=Lookup[{##2},"Method","GET"]},
                                    URLFetch[#1, {"StatusCode", "Content"}, "Method" -> method, "Parameters" -> params, "Body" -> Lookup[{##2},"BodyData",{}], "MultipartElements" -> Lookup[{##2},"MultipartData",{}], "VerifyPeer" -> True, "CredentialsProvider" -> None]
                                    ]&),
        "Gets"                    -> {"GetTweet","LastTweet","FollowerIDs","FriendIDs","UserData","UserMentions","UserReplies","FollowerMentionNetwork",
            "FriendMentionNetwork","FollowerReplyToNetwork","FriendReplyToNetwork","FriendNetwork","FollowerNetwork","Friends","Followers",
            "RateLimit","UserIDSearch","SearchNetwork","SearchReplyToNetwork","SearchMentionNetwork","UserHashtags","TweetSearch",
            "TweetEventSeries","TweetTimeline","TweetList"},
        "Posts"                   -> {"Tweet","Retweet"},
        "RawGets"                 -> {"RawMentionsTimeline","RawUserTimeline","RawHomeTimeline","RawRetweetTimeline",
                    "RawStatus","RawRetweets","RawRetweeterIDs","RawTweetSearch",
                    "RawDirectMessages","RawDirectMessagesSent","RawDirectMessage",
                    "RawNoRetweetUserIDs", "RawFriendIDs", "RawFollowerIDs",
                    "RawMyFriendship", "RawIncomingFriendships",
                    "RawOutgoingFriendships", "RawFriendship", "RawFriends",
                    "RawFollowers","RawUserSettings","RawVerifyCredentials","RawBlockList","RawBlockIDs",
                    "RawUsers","RawUser","RawUserSearch","RawContributees","RawContributors",
                    "RawSuggestedUsers","RawSuggestedUserCategories","RawSuggestedUserStatuses","RawFavorites","RawAccountStatus"},
        "RawPosts"                -> {
                    "RawDeleteTweet",
                    "RawUpdate","RawRetweet","RawMediaUpload","RawUpload",
                    "RawDeleteDirectMessage","RawSendDirectMessage",
                    "RawUpdateFollowing", "RawStopFollowing", "RawStartFollowing",
                    "RawSetUserSettings","RawUpdateDevice","RawUpdateProfile","RawUpdateBackgroundImage",
                    "RawUpdateProfileColors",
                    (* "RawUpdateProfileImage",*)"RawCreateBlock","RawRemoveBlock","RawAddFavorite","RawRemoveFavorite"},
        "LogoutURL"               -> "https://twitter.com/logout",
        "Information"             -> "A service for sending and receiving tweets from a Twitter account"
    }
]

$rateLimitFlagged = False

(* a function for importing the raw data - usually json or xml - from the service *)
twitterimport[$Failed]:= Throw[$Failed]
twitterimport[response_]:= Block[{
    statusCode = response[[1]], result = response[[2]], data, errorCode, error},

    Switch[statusCode,
        200,
            data = Developer`ReadRawJSONString @ result
        ,

        401,
            Message[ServiceExecute::unauth];
            Throw[$Failed]
        ,

        404,
            Message[ServiceExecute::nouser];
            Throw[$Failed]
        ,

        _,
            If[ result =!= "",
                data = Developer`ReadRawJSONString @ result;
                error = Lookup[data, "errors", Lookup[data, "error"]];
                error = If[ListQ[error], First[error], error];
                If[ AssociationQ[error],
                    errorCode = Lookup[error, "code", -1];
                    error = Lookup[error, "message", Missing["NotAvailable"]]
                ,
                    errorCode = -1;
                    error = Missing["NotAvailable"]
                ];

                Which[
                    MatchQ[errorCode, 88], (* Rate limit exceeded *)
                        $rateLimitFlagged = True
                    ,

                    StringQ[error] && StringLength[StringTrim[error]] > 0,
                        Message[ServiceExecute::serrormsg, error]
                    ,

                    True,
                        Message[ServiceExecute::errimp]
                ]
                ,
                Message[ServiceExecute::errimp]
            ];

            Throw[$Failed]
    ]
]

(*** Raw ***)

(** Timelines **)
twitterdata["RawMentionsTimeline"] = {
        "URL"                -> "https://api.twitter.com/1.1/statuses/mentions_timeline.json",
        "Parameters"         -> {"count","since_id","max_id","trim_user","contributor_details","include_entities","tweet_mode"},
        "RequiredParameters" -> {},
        "HTTPSMethod"        -> "GET",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawUserTimeline"] = {
        "URL"                -> "https://api.twitter.com/1.1/statuses/user_timeline.json",
        "Parameters"         -> {"user_id","screen_name","count","since_id","max_id","trim_user","exclude_replies","contributor_details","include_rts","tweet_mode"},
        "RequiredParameters" -> {"user_id"|"screen_name"},
        "HTTPSMethod"        -> "GET",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawHomeTimeline"] = {
        "URL"                -> "https://api.twitter.com/1.1/statuses/home_timeline.json",
        "Parameters"         -> {"count","since_id","max_id","trim_user","exclude_replies","contributor_details","include_entities","tweet_mode"},
        "HTTPSMethod"        -> "GET",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawRetweetTimeline"] = {
        "URL"                -> "https://api.twitter.com/1.1/statuses/retweets_of_me.json",
        "Parameters"         -> {"count","since_id","max_id","trim_user","include_entities","include_user_entities","tweet_mode"},
        "HTTPSMethod"        -> "GET",
        "ResultsFunction"    -> twitterimport
    }

(** Tweets **)

twitterdata["RawRetweets"] = {
        "URL"                -> "https://api.twitter.com/1.1/statuses/retweets.json",
        "Parameters"         -> {"id","count","trim_user","tweet_mode"},
        "RequiredParameters" -> {"id"},
        "HTTPSMethod"        -> "GET",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawStatus"] = {
        "URL"                -> "https://api.twitter.com/1.1/statuses/show.json",
        "Parameters"         -> {"id","include_my_retweet","trim_user","include_entities","include_ext_alt_text","tweet_mode"},
        "RequiredParameters" -> {"id"},
        "HTTPSMethod"        -> "GET",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawDeleteTweet"] = {
        "URL"                -> (ToString@StringForm["https://api.twitter.com/1.1/statuses/destroy/`1`.json", #]&),
        "PathParameters"     -> {"id"},
        "BodyData"           -> {"trim_user"},
        "RequiredParameters" -> {"id","trim_user"(* Should not be required but URLFetch fails when the BodyData is empty *)},
        "HTTPSMethod"        -> "POST",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawUpdate"] = {
        "URL"                -> "https://api.twitter.com/1.1/statuses/update.json",
        "BodyData"           -> {"status","in_reply_to_status_id","lat","long","place_id","display_coordinates","trim_user","media_ids"},
        "RequiredParameters" -> {"status"},
        "HTTPSMethod"        -> "POST",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawRetweet"] = {
        "URL"                -> (ToString@StringForm["https://api.twitter.com/1.1/statuses/retweet/`1`.json", #]&),
        "PathParameters"     -> {"id"},
        "BodyData"           -> {"trim_user"},
        "RequiredParameters" -> {"id","trim_user"(* Should not be required but URLFetch fails when the BodyData is empty *)},
        "HTTPSMethod"        -> "POST",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawMediaUpload"] = {
        "URL"                -> "https://api.twitter.com/1.1/statuses/update_with_media.json",
        "MultipartData"      -> {{"media[]","image/jpeg"},{"status","text/plain"},
            {"possibly_sensitive","text/plain"},{"in_reply_to_status_id","text/plain"},
            {"lat","text/plain"},{"long","text/plain"},{"place_id","text/plain"},{"display_coordinates","text/plain"} },
        "RequiredParameters" -> {"media[]","status"},
        "HTTPSMethod"        -> "POST",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawUpload"] = {
        "URL"                -> "https://upload.twitter.com/1.1/media/upload.json",
        "MultipartData"      -> {{"media","image/jpeg"}},
        "RequiredParameters" -> {"media"},
        "HTTPSMethod"        -> "POST",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawRetweeterIDs"] = {
        "URL"                -> "https://api.twitter.com/1.1/statuses/retweeters/ids.json",
        "Parameters"         -> {"id","cursor","stringify_ids","tweet_mode"},
        "RequiredParameters" -> {"id"},
        "HTTPSMethod"        -> "GET",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawTweetSearch"] = {
        "URL"                -> "https://api.twitter.com/1.1/search/tweets.json",
        "Parameters"         -> {"q","geocode","lang","locale","result_type","count","until","since_id","max_id","include_entities","tweet_mode"},
        "RequiredParameters" -> {"q"},
        "HTTPSMethod"        -> "GET",
        "ResultsFunction"    -> twitterimport
    }

(* Stream API requires server support *)

(** Direct messages **)


twitterdata["RawDirectMessages"] = {
        "URL"                -> "https://api.twitter.com/1.1/direct_messages.json",
        "Parameters"         -> {"count","since_id","max_id","include_entities","skip_status","tweet_mode"},
        "HTTPSMethod"        -> "GET",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawDirectMessagesSent"] = {
        "URL"                -> "https://api.twitter.com/1.1/direct_messages/sent.json",
        "Parameters"         -> {"count","since_id","max_id","include_entities","page","tweet_mode"},
        "HTTPSMethod"        -> "GET",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawDirectMessage"] = {
        "URL"                -> "https://api.twitter.com/1.1/direct_messages/show.json",
        "Parameters"         -> {"id","tweet_mode"},
        "RequiredParameters" -> {"id"},
        "HTTPSMethod"        -> "GET",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawDeleteDirectMessage"] = {
        "URL"                -> "https://api.twitter.com/1.1/direct_messages/destroy.json",
        "BodyData"           -> {"id","include_entities","tweet_mode"},
        "RequiredParameters" -> {"id"},
        "HTTPSMethod"        -> "POST",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawSendDirectMessage"] = {
        "URL"                -> "https://api.twitter.com/1.1/direct_messages/new.json",
        "BodyData"           -> {"user_id","screen_name","text"},
        "RequiredParameters" -> {"user_id"|"screen_name","text"},
        "HTTPSMethod"        -> "POST",
        "ResultsFunction"    -> twitterimport
    }

(* Friends and Followers *)

twitterdata["RawNoRetweetUserIDs"] = {
        "URL"                -> "https://api.twitter.com/1.1/friendships/no_retweets/ids.json",
        "Parameters"         -> {"stringify_ids","tweet_mode"},
        "HTTPSMethod"        -> "GET",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawFriendIDs"] = { (* who the authenticated user is following *)
        "URL"                -> "https://api.twitter.com/1.1/friends/ids.json",
        "Parameters"         -> {"user_id","screen_name","cursor","stringify_ids","count","tweet_mode"},
        "HTTPSMethod"        -> "GET",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawFollowerIDs"] = { (* who is following the specified user *)
        "URL"                -> "https://api.twitter.com/1.1/followers/ids.json",
        "Parameters"         -> {"user_id","screen_name","cursor","stringify_ids","count","tweet_mode"},
        "RequiredParameters" -> {"user_id"|"screen_name"},
        "HTTPSMethod"        -> "GET",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawMyFriendship"] = { (* for the specified user, takes a comma separated list *)
        "URL"                -> "https://api.twitter.com/1.1/friendships/lookup.json",
        "Parameters"         -> {"user_id","screen_name","tweet_mode"},
        "RequiredParameters" -> {"user_id"|"screen_name"},
        "HTTPSMethod"        -> "GET",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawIncomingFriendships"] = { (* for the specified user *)
        "URL"                -> "https://api.twitter.com/1.1/friendships/incoming.json",
        "Parameters"         -> {"cursor","stringify_ids","tweet_mode"},
        "HTTPSMethod"        -> "GET",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawOutgoingFriendships"] = { (* for the specified user *)
        "URL"                -> "https://api.twitter.com/1.1/friendships/outgoing.json",
        "Parameters"         -> {"cursor","stringify_ids","tweet_mode"},
        "HTTPSMethod"        -> "GET",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawStartFollowing"] = {
        "URL"                -> "https://api.twitter.com/1.1/friendships/create.json",
        "BodyData"           -> {"user_id","screen_name","follow"},
        "RequiredParameters" -> {"user_id"|"screen_name"},
        "HTTPSMethod"        -> "POST",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawStopFollowing"] = {
        "URL"                -> "https://api.twitter.com/1.1/friendships/destroy.json",
        "BodyData"           -> {"user_id","screen_name"},
        "RequiredParameters" -> {"user_id"|"screen_name"},
        "HTTPSMethod"        -> "POST",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawUpdateFollowing"] = { (* turn devive or retweet notifications on/off *)
        "URL"                -> "https://api.twitter.com/1.1/friendships/update.json",
        "BodyData"           -> {"user_id","screen_name","device","retweets"},
        "RequiredParameters" -> {"user_id"|"screen_name"},
        "HTTPSMethod"        -> "POST",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawFriendship"] = { (* between any two users *)
        "URL"                -> "https://api.twitter.com/1.1/friendships/show.json",
        "Parameters"         -> {"source_id","source_screen_name","target_id","target_screen_name","tweet_mode"},
        "RequiredParameters" -> {"source_id"|"source_screen_name","target_id"|"target_screen_name"},
        "HTTPSMethod"        -> "GET",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawFriends"] = { (* who the specified user is following *)
        "URL"                -> "https://api.twitter.com/1.1/friends/list.json",
        "Parameters"         -> {"user_id","screen_name","cursor","skip_status","include_user_entities","count","tweet_mode"},
        "RequiredParameters" -> {"user_id"|"screen_name"},
        "HTTPSMethod"        -> "GET",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawFollowers"] = { (* who is following the specified user *)
        "URL"                -> "https://api.twitter.com/1.1/followers/list.json",
        "Parameters"         -> {"user_id","screen_name","cursor","skip_status","include_user_entities","count","tweet_mode"},
        "RequiredParameters" -> {"user_id"|"screen_name"},
        "HTTPSMethod"        -> "GET",
        "ResultsFunction"    -> twitterimport
    }

(** Users **)
twitterdata["RawUserSettings"] = {
        "URL"                -> "https://api.twitter.com/1.1/account/settings.json",
        "HTTPSMethod"        -> "GET",
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawVerifyCredentials"] = {
        "URL"                -> "https://api.twitter.com/1.1/account/verify_credentials.json",
        "HTTPSMethod"        -> "GET",
        "Parameters"         -> {"include_entities","skip_status"},
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawSetUserSettings"] = {
        "URL"                -> "https://api.twitter.com/1.1/account/settings.json",
        "HTTPSMethod"        -> "POST",
        "BodyData"           -> {"trend_location_woeid","sleep_time_enabled","start_sleep_time","end_sleep_time","time_zone","lang"},
        "RequiredParameters" -> {"trend_location_woeid"|"sleep_time_enabled"|"start_sleep_time"|"end_sleep_time"|"time_zone"|"lang"},
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawUpdateDevice"] = {
        "URL"                -> "https://api.twitter.com/1.1/account/settings.json",
        "HTTPSMethod"        -> "POST",
        "BodyData"           -> {"device","include_entities"},
        "RequiredParameters" -> {"device"},
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawUpdateProfile"] = {
        "URL"                -> "https://api.twitter.com/1.1/account/update_profile.json",
        "HTTPSMethod"        -> "POST",
        "BodyData"           -> {"name","url","location","description","include_entities","skip_status"},
        "RequiredParameters" -> {"name"|"url"|"location"|"description"|"include_entities"|"skip_status"},
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawUpdateBackgroundImage"] = {
        "URL"                -> "https://api.twitter.com/1.1/account/update_profile_background_image.json",
        "HTTPSMethod"        -> "POST",
        "MultipartData"      -> {{"image","image/jpeg"},{"tile","text/plain"},{"status","text/plain"},{"include_entities","text/plain"},
            {"include_entities","text/plain"},{"skip_status","text/plain"},{"use","text/plain"}},
        "RequiredParameters" -> {"image"|"tile"|"use"},
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawUpdateProfileColors"] = {
        "URL"                -> "https://api.twitter.com/1.1/account/update_profile_colors.json",
        "HTTPSMethod"        -> "POST",
        "BodyData"           -> {"profile_background_color","profile_link_color","profile_sidebar_border_color",
            "profile_sidebar_fill_color","profile_text_color","include_entities","skip_status"},
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawUpdateProfileImage"] = {
        "URL"                -> "https://api.twitter.com/1.1/account/update_profile.json",
        "HTTPSMethod"        -> "POST",
        "BodyData"           -> {"image","include_entities","skip_status"},
       (* "MultipartData"        -> {{"image","image/jpeg"},{"include_entities","text/plain"},{"skip_status","text/plain"}},*)
        "RequiredParameters" -> {"image"},
        "ResultsFunction"    -> twitterimport
    }


(* Blocks *)
twitterdata["RawBlockList"] = {
        "URL"                -> "https://api.twitter.com/1.1/blocks/list.json",
        "HTTPSMethod"        -> "GET",
        "Parameters"         -> {"include_entities","skip_status","cursor"},
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawBlockIDs"] = {
        "URL"                -> "https://api.twitter.com/1.1/blocks/ids.json",
        "HTTPSMethod"        -> "GET",
        "Parameters"         -> {"stringify_ids","cursor"},
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawCreateBlock"] = {
        "URL"                -> "https://api.twitter.com/1.1/blocks/create.json",
        "HTTPSMethod"        -> "POST",
        "BodyData"           -> {"screen_name","user_id","include_entities","skip_status"},
        "RequiredParameters" -> {"screen_name"|"user_id"},
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawRemoveBlock"] = {
        "URL"                -> "https://api.twitter.com/1.1/blocks/destroy.json",
        "HTTPSMethod"        -> "POST",
        "BodyData"           -> {"screen_name","user_id","include_entities","skip_status"},
        "RequiredParameters" -> {"screen_name"|"user_id"},
        "ResultsFunction"    -> twitterimport
    }
(* Users *)
twitterdata["RawUsers"] = { (* comma separated list of users *)
        "URL"                -> "https://api.twitter.com/1.1/users/lookup.json",
        "HTTPSMethod"        -> "GET",
        "Parameters"         -> {"screen_name","user_id","include_entities","tweet_mode"},
        "RequiredParameters" -> {"screen_name"|"user_id"},
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawUser"] = { (* a single user, with more information *)
        "URL"                -> "https://api.twitter.com/1.1/users/show.json",
        "HTTPSMethod"        -> "GET",
        "Parameters"         -> {"screen_name","user_id","include_entities","tweet_mode"},
        "RequiredParameters" -> {"screen_name"|"user_id"},
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawUserSearch"] = {
        "URL"                -> "https://api.twitter.com/1.1/users/search.json",
        "HTTPSMethod"        -> "GET",
        "Parameters"         -> {"q","page","count","include_entities","tweet_mode"},
        "RequiredParameters" -> {"q"},
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawContributees"] = {
        "URL"                -> "https://api.twitter.com/1.1/users/contributees.json",
        "HTTPSMethod"        -> "GET",
        "Parameters"         -> {"screen_name","user_id","include_entities","skip_status","tweet_mode"},
        "RequiredParameters" -> {"screen_name"|"user_id"},
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawContributors"] = {
        "URL"                -> "https://api.twitter.com/1.1/users/contributors.json",
        "HTTPSMethod"        -> "GET",
        "Parameters"         -> {"screen_name","user_id","include_entities","skip_status","tweet_mode"},
        "RequiredParameters" -> {"screen_name"|"user_id"},
        "ResultsFunction"    -> twitterimport
    }

(* profile banners omitted for now *)

(** Suggested Users **)
twitterdata["RawSuggestedUsers"] = {
        "URL"                -> (ToString@StringForm["https://api.twitter.com/1.1/users/suggestions/`1`.json", #]&),
        "HTTPSMethod"        -> "GET",
        "PathParameters"     -> {"slug"},
        "Parameters"         -> {"lang"},
        "RequiredParameters" -> {"slug"},
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawSuggestedUserCategories"] = {
        "URL"                -> "https://api.twitter.com/1.1/users/suggestions.json",
        "HTTPSMethod"        -> "GET",
        "Parameters"         -> {"lang"},
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawSuggestedUserStatuses"] = {
        "URL"                -> (ToString@StringForm["https://api.twitter.com/1.1/users/suggestions/`1`/members.json", #]&),
        "HTTPSMethod"        -> "GET",
        "PathParameters"     -> {"slug"},
        "Parameters"         -> {"lang"},
        "RequiredParameters" -> {"slug"},
        "ResultsFunction"    -> twitterimport
    }

(** Favorites **)
twitterdata["RawFavorites"] = {
        "URL"                -> "https://api.twitter.com/1.1/favorites/list.json",
        "HTTPSMethod"        -> "GET",
        "Parameters"         -> {"screen_name","user_id","count","include_entities","since_id","max_id"},
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawRemoveFavorite"] = {
        "URL"                -> "https://api.twitter.com/1.1/favorites/destroy.json",
        "HTTPSMethod"        -> "POST",
        "BodyData"           -> {"id","include_entities"},
        "RequiredParameters" -> {"id"},
        "ResultsFunction"    -> twitterimport
    }

twitterdata["RawAddFavorite"] = {
        "URL"                -> "https://api.twitter.com/1.1/favorites/create.json",
        "HTTPSMethod"        -> "POST",
        "BodyData"           -> {"id","include_entities"},
        "RequiredParameters" -> {"id"},
        "ResultsFunction"    -> twitterimport
    }



(*** App ***)
twitterdata["RawAccountStatus"] = {
        "URL"                -> "https://api.twitter.com/1.1/application/rate_limit_status.json",
        "HTTPSMethod"        -> "GET",
        "ResultsFunction"    -> twitterimport
    }

(** Lists **)
(** Saved Searches **)
(** Places and Geo **)
(** Trends **)
(** Spam reporting **)

twitterdata["icon"]:=bird

twitterdata[___]:=$Failed

(****** Cooked Properties ******)

getAuthenticatedTwitterID[id_]:=If[ValueQ[authtwitterid[id]]&&authtwitterid[id]=!=$Failed,
    authtwitterid[id],
    authtwitterid[id]=Block[{rawdata=OAuthClient`rawoauthdata[id,"RawVerifyCredentials"],importeddata},
    importeddata=twitterimport[rawdata];
        Lookup[importeddata,"id_str",Throw[$Failed]]
    ]
]

Twittercookeddata[args___] := Module[
    {res = Catch[twittercookeddata[args]]},

    If[ $rateLimitFlagged,
        $rateLimitFlagged = False;
        Message[ServiceExecute::rlim]
    ];

    If[ !FailureQ[res],
        res
        ,
        Throw[$Failed]
    ]
]

twittercookeddata[prop:("FollowerIDs"|"Followers"|"FriendIDs"|"Friends"),id_,args_]:=Module[
    {invalidParameters,rawdata, newargs=KeyMap[ToString]@Association@args,params={},ids,prop1,limit,user,name,data},

    invalidParameters = Select[Keys[newargs],!MemberQ[{"UserID","Username","MaxItems","Elements"},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Twitter"]&/@invalidParameters;
        Throw[$Failed]
    ];

    prop1 = Switch[prop,
        "FollowerIDs"|"Followers","RawFollowerIDs",
        "FriendIDs"|"Friends","RawFriendIDs"
    ];

    If[ KeyExistsQ[newargs, "MaxItems"],
        limit = Lookup[newargs, "MaxItems"];
        If[ !(IntegerQ[limit] && 5001>limit>0),
            Message[ServiceExecute::nval,MaxItems,"Twitter"];
            Throw[$Failed]
        ];
        AppendTo[params,"count"->ToString@limit]
    ];

    If[ KeyExistsQ[newargs, "UserID"],
        user = ToString@Lookup[newargs, "UserID"];
        If[ !StringMatchQ[user, DigitCharacter..],
            Message[ServiceExecute::nval,"UserID","Twitter"];
            Throw[$Failed]
        ];
        AppendTo[params,"user_id"->user]
        ,
        If[ KeyExistsQ[newargs, "Username"],
            name = Lookup[newargs, "Username"];
            If[ !StringQ[name],
                Message[ServiceExecute::nval,"Username","Twitter"];
                Throw[$Failed]
            ];
            AppendTo[params,"screen_name"->name]
            ,
            AppendTo[params,"user_id"->getAuthenticatedTwitterID[id]]
        ];
    ];

    rawdata = OAuthClient`rawoauthdata[id,prop1,params];
    data = twitterimport[rawdata];

    ids = (TwitterToString /@ Lookup[data, "ids", {}]);
    If[ MatchQ[prop,("Followers"|"Friends")],
        TwitterGetscreennames[id,ids],
        ids
        ]

]

twittercookeddata["GetTweet",id_,args_]:=Module[
    {invalidParameters, rawdata, params, twid, data},

    invalidParameters = Select[Keys[args],!MemberQ[{"TweetID","Elements","ShowThumbnails","MediaResolution","ShowIDs"},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Twitter"]&/@invalidParameters;
        Throw[$Failed]
    ];

    If[ KeyExistsQ[args, "TweetID"],
        twid = ToString@Lookup[args, "TweetID"];
        If[ !StringMatchQ[twid, DigitCharacter..],
            Message[ServiceExecute::nval,"TweetID","Twitter"];
            Throw[$Failed]
        ]
        ,
        Message[ServiceExecute::nparam,"TweetID"];
        Throw[$Failed]
    ];

    params = TwitterToString /@ Association[args];

    If[ !KeyExistsQ[params, "Elements"] || FreeQ[{"Text", "Images", "Data", "Default", Default, "FullData"}, params["Elements"]],
        params["Elements"] = Default;
    ];

    rawdata=OAuthClient`rawoauthdata[id,"RawStatus",{"id"->twid, "tweet_mode"->"extended"}];

    data = TwitterFormatByElementType[{twitterimport[rawdata]},params];
    If[ Length[data] > 0,
        Switch[params["Elements"],
            "Text",
                First[data]
            ,

            "Images",
                First[data]
            ,

            "FullData"|"Default"|Default|"Data"|_,
                Dataset[First[data]]
        ]
    ,
        data
    ]
]

twittercookeddata[prop:("TweetEventSeries"|"TweetTimeline"|"TweetList"),id_,args_]:=Module[
    {invalidParameters,newargs=KeyMap[ToString]@Association@args,params=<|"tweet_mode"->"extended"|>,data,dates,user,name,limit,query,restype,since,max,tweets},

    invalidParameters = Select[Keys[newargs],!MemberQ[{"UserID","Username","MaxItems","SinceID","MaxID","Elements","ShowThumbnails","MediaResolution","ShowIDs"},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Twitter"]&/@invalidParameters;
        Throw[$Failed]
    ];

    If[ KeyExistsQ[newargs, "MaxItems"],
        limit = Lookup[newargs, "MaxItems"];
        If[ !(IntegerQ[limit] && 5001>limit>0),
            Message[ServiceExecute::nval,MaxItems,"Twitter"];
            Throw[$Failed]
        ];
        AppendTo[params,"count"->ToString@limit],
        AppendTo[params,"count"->"20"]
    ];

    If[ KeyExistsQ[newargs, "UserID"],
        user = ToString@Lookup[newargs, "UserID"];
        If[ !StringMatchQ[user, DigitCharacter..],
            Message[ServiceExecute::nval,"UserID","Twitter"];
            Throw[$Failed]
        ];
        AppendTo[params,"user_id"->user]
        ,
        If[ KeyExistsQ[newargs, "Username"],
            name = Lookup[newargs, "Username"];
            If[ !StringQ[name],
                Message[ServiceExecute::nval,"Username","Twitter"];
                Throw[$Failed]
            ];
            AppendTo[params,"screen_name"->name]
            ,
            AppendTo[params,"user_id"->getAuthenticatedTwitterID[id]]
        ];
    ];

    If[ KeyExistsQ[newargs, "Query"],
        query = Lookup[newargs, "Query"];
        If[ !StringQ[query],
            Message[ServiceExecute::nval,"Query","Twitter"];
            Throw[$Failed]
        ];
        AppendTo[params,"q"->query]
    ];

    If[ KeyExistsQ[newargs, "ResultType"],
        restype = Lookup[newargs, "ResultType"];
        If[ !MatchQ[restype, "Popular"|"Recent"],
            Message[ServiceExecute::nval,"ResultType","Twitter"];
            Throw[$Failed]
        ];
        AppendTo[params,"result_type"->restype]
    ];

    If[ KeyExistsQ[newargs, "SinceID"],
        since = ToString@Lookup[newargs, "SinceID"];
        If[ !StringMatchQ[since, DigitCharacter..],
            Message[ServiceExecute::nval,"SinceID","Twitter"];
            Throw[$Failed]
        ];
        AppendTo[params,"since_id"->since]
    ];

    If[ KeyExistsQ[newargs, "MaxID"],
        max = ToString@Lookup[newargs, "MaxID"];
        If[ !StringMatchQ[max, DigitCharacter..],
            Message[ServiceExecute::nval,"MaxID","Twitter"];
            Throw[$Failed]
        ];
        AppendTo[params,"max_id"->max]
    ];

    If[ prop =!= "TweetList" || !KeyExistsQ[newargs, "Elements"] || FreeQ[{"Text", "Images", "Data", "Default", Default, "FullData"}, newargs["Elements"]],
        AppendTo[params,"Elements"->Default],
        AppendTo[params,"Elements"->newargs["Elements"]]
    ];

    limit= FromDigits[params["count"]];
    data = TwitterPaginationCalls[id,"RawUserTimeline",params,200];
    Switch[prop,
        "TweetList",
            data = TwitterFormatByElementType[data,params];
            If[ Length[data] > 0,
                Switch[params["Elements"],
                    "Text",
                        data
                    ,

                    "Images",
                        data
                    ,

                    "FullData"|"Default"|Default|"Data"|_,
                        Dataset[data]
                ]
            ,
                data
            ],
        "TweetEventSeries",
            EventSeries[Lookup[MapAt[TwitterReadDate, data, {All, Key["created_at"]}], {"created_at", "full_text"}]],
        "TweetTimeline",
            {dates, tweets} = Transpose[Lookup[Reverse@MapAt[TwitterReadDate, data, {All, Key["created_at"]}], {"created_at", "full_text"}]];
            TwitterEventTimeline[tweets, dates]
    ]
]

twittercookeddata["LastTweet",id_,args_]:=Module[
    {invalidParameters,newargs=KeyMap[ToString]@Association@args,params=<|"tweet_mode"->"extended","count"->"1"|>,rawdata,data,user,name},

    invalidParameters = Select[Keys[newargs],!MemberQ[{"UserID","Username","Elements"},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Twitter"]&/@invalidParameters;
        Throw[$Failed]
    ];

    If[ KeyExistsQ[newargs, "UserID"],
        user = ToString@Lookup[newargs, "UserID"];
        If[ !StringMatchQ[user, DigitCharacter..],
            Message[ServiceExecute::nval,"UserID","Twitter"];
            Throw[$Failed]
        ];
        AppendTo[params,"user_id"->user]
        ,
        If[ KeyExistsQ[newargs, "Username"],
            name = Lookup[newargs, "Username"];
            If[ !StringQ[name],
                Message[ServiceExecute::nval,"Username","Twitter"];
                Throw[$Failed]
            ];
            AppendTo[params,"screen_name"->name]
            ,
            AppendTo[params,"user_id"->getAuthenticatedTwitterID[id]]
        ];
    ];

    rawdata=OAuthClient`rawoauthdata[id,"RawUserTimeline",Normal@params];
    data = twitterimport[rawdata];
    If[ Length[data] > 0,
        First[data]["full_text"],
        Missing["NotAvailable"]
    ]
]

twittercookeddata["TweetSearch",id_,args_]:=Module[
    {invalidParameters,newargs=KeyMap[ToString]@Association@args,params=<|"tweet_mode"->"extended"|>,limit,query,geocode,radius,position,items},

    invalidParameters = Select[Keys[newargs],!MemberQ[{"GeoLocation","Radius","Language","MaxItems","Elements","Query","ResultType"},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Twitter"]&/@invalidParameters;
        Throw[$Failed]
    ];

    If[ KeyExistsQ[newargs, "Query"],
        query = Lookup[newargs, "Query"];
        If[ !StringQ[query],
            Message[ServiceExecute::nval,"Query","Twitter"];
            Throw[$Failed]
        ];
        AppendTo[params,"q"->query],        
        Message[ServiceExecute::nparam,"Query"];
        Throw[$Failed]
    ];

    If[ !KeyExistsQ[newargs,"Elements"],
        AppendTo[params,"Elements"->Default],
        AppendTo[params,"Elements"->newargs["Elements"]]
    ];

    If[ KeyExistsQ[newargs, "MaxItems"],
        limit = Lookup[newargs, "MaxItems"];
        If[ !(IntegerQ[limit] && 5001>limit>0),
            Message[ServiceExecute::nval,MaxItems,"Twitter"];
            Throw[$Failed]
        ];
        AppendTo[params,"count"->ToString@limit],
        AppendTo[params,"count"->"200"]
    ];

    If[ KeyExistsQ[newargs, "GeoLocation"],
        position = Lookup[newargs,"GeoLocation"];

        If[ MatchQ[GeoPosition[position], GeoPosition[{_?NumericQ, _?NumericQ}]] || MatchQ[GeoPosition[position], GeoPosition[{_?NumericQ, _?NumericQ, _?NumericQ}]],
            If[ KeyExistsQ[newargs, "Radius"],
                If[ Quiet@QuantityQ[newargs["Radius"]],
                    If[ CompatibleUnitQ[newargs["Radius"], "Kilometers"],
                        newargs["Radius"] = UnitConvert[newargs["Radius"], "Kilometers"];
                        radius = StringReplace[ToString@N@QuantityMagnitude[newargs["Radius"]],"." ~~ EndOfString -> ""] <> "km";
                    ,
                        Message[ServiceExecute::nval,"Radius","Twitter"];
                        Throw[$Failed]
                    ];
                ,
                    Message[ServiceExecute::nval,"Radius","Twitter"];
                    Throw[$Failed]
                ];
            ,
                radius = "30km";
            ];
            geocode = ToString[QuantityMagnitude@Latitude[position]] <> "," <> ToString[QuantityMagnitude@Longitude[position]] <> "," <> radius;
            AppendTo[params,"geocode"->geocode];
        ,
            Message[ServiceExecute::nval,"GeoPosition","Twitter"];
            Throw[$Failed]
        ];
    ];

    If[ KeyExistsQ[newargs, "Language"],
        If[ Head[newargs["Language"]] === Entity,
            If[ KeyExistsQ[TwitterEntityToLanguageCode, newargs["Language"]],
                AppendTo[params,"lang"-> (newargs["Language"] /. TwitterEntityToLanguageCode)];
            ,
                Message[ServiceExecute::nval,"Language","Twitter"];
                Throw[$Failed]
            ];
        ,
            Message[ServiceExecute::nval,"Language","Twitter"];
            Throw[$Failed]
        ];
    ];

    If[ KeyExistsQ[newargs, "ResultType"],
        newargs["ResultType"] = ToLowerCase[ToString[newargs["ResultType"]]];
        Switch[ newargs["ResultType"],
            "recent"|"popular",
                AppendTo[params,"result_type"->newargs["ResultType"]];
            ,
            _,
                Message[ServiceExecute::nval,"ResultType","Twitter"];
                Throw[$Failed]
        ];
    ];

    items = TwitterPaginationCalls[id,"RawTweetSearch",params,100];
    items = MapAt[TwitterReadDate, items, {All, Key["created_at"]}];

    Dataset[items][All, <|"ID"->"id","Text"->"full_text","CreationDate"->"created_at","Entities"->"entities","Language"->"lang","FavoriteCount"->"favorite_count","RetweetCount"->"retweet_count"|>]
]


twittercookeddata["UserData",id_,args_]:=Module[
    {invalidParameters,newargs=KeyMap[ToString]@Association@args,params=<|"tweet_mode"->"extended"|>,user,name,rawdata},

    invalidParameters = Select[Keys[newargs],!MemberQ[{"UserID","Username","Elements"},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Twitter"]&/@invalidParameters;
        Throw[$Failed]
    ];

    If[ KeyExistsQ[newargs, "UserID"],
        user = Lookup[newargs, "UserID"];
        Which[
            ListQ[user],
                user = ToString /@ user;
                If[ !AllTrue[user, StringMatchQ[DigitCharacter..]],
                    Message[ServiceExecute::nval,"UserID","Twitter"];
                    Throw[$Failed]
                ],
            StringMatchQ[ToString@user, DigitCharacter..],
                user = {ToString@user},
            True,
                Message[ServiceExecute::nval,"UserID","Twitter"];
                Throw[$Failed]
        ];
        name = Missing["NotFound"]
        ,
        If[ KeyExistsQ[newargs, "Username"],
            name = Lookup[newargs, "Username"];
            Which[
                ListQ[name],
                    name = ToString /@ name,
                StringQ[name],
                    name = {name},
                True,
                    Message[ServiceExecute::nval,"Username","Twitter"];
                    Throw[$Failed]
            ];
            user = Missing["NotFound"]
            ,
            AppendTo[params,"user_id"->getAuthenticatedTwitterID[id]]
        ]
    ];

    Which[
        KeyExistsQ[params, "user_id"],
            rawdata=twitterimport[OAuthClient`rawoauthdata[id,"RawUsers",Normal@params]],
        MissingQ[user] && !MissingQ[name],
            name = StringRiffle[#, ","]& /@ Partition[name,UpTo[100]];
            rawdata = Flatten@((twitterimport[OAuthClient`rawoauthdata[id,"RawUsers",Normal@Append[params,"screen_name"->#]]])&/@name),
        MissingQ[name] && !MissingQ[user],
            user = StringRiffle[#, ","]& /@ Partition[user,UpTo[100]];
            rawdata = Flatten@((twitterimport[OAuthClient`rawoauthdata[id,"RawUsers",Normal@Append[params,"user_id"->#]]])&/@user)
    ];

    (MapAt[DateObject[StringReplace[#,"+0000"-> ""],TimeZone->0]&,Key["CreationDate"]]@MapAt[Replace[""->Missing["NotAvailable"]],Key["Location"]]@KeyMap[#/.{"CreatedAt"-> "CreationDate","StatusesCount"-> "TweetsCount"}&]@KeyMap[TwitterCamelCase][KeyTake[#,{"id","screen_name","name","location","favourites_count","followers_count","friends_count","statuses_count","created_at"}]])& /@ rawdata
]

twittercookeddata["UserHashtags",id_,args_]:=Module[
    {invalidParameters,newargs=KeyMap[ToString]@Association@args,params=<|"tweet_mode"->"extended"|>,res,user,name,tmp,limit,rawdata,expectedSize=0,retries=0,maxRetries=10,size=0,result={}},

    invalidParameters = Select[Keys[newargs],!MemberQ[{"UserID","Username","MaxItems","Elements"},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Twitter"]&/@invalidParameters;
        Throw[$Failed]
    ];

    If[ KeyExistsQ[newargs, "MaxItems"],
        limit = Lookup[newargs, "MaxItems"];
        If[ !(IntegerQ[limit] && 200>limit>0),
            Message[ServiceExecute::nval,MaxItems,"Twitter"];
            Throw[$Failed]
        ];
        AppendTo[params,"count"->ToString@limit],
        AppendTo[params,"count"->"20"]
    ];

    If[ KeyExistsQ[newargs, "UserID"],
        user = ToString@Lookup[newargs, "UserID"];
        If[ !StringMatchQ[user, DigitCharacter..],
            Message[ServiceExecute::nval,"UserID","Twitter"];
            Throw[$Failed]
        ];
        AppendTo[params,"user_id"->user]
        ,
        If[ KeyExistsQ[newargs, "Username"],
            name = Lookup[newargs, "Username"];
            If[ !StringQ[name],
                Message[ServiceExecute::nval,"Username","Twitter"];
                Throw[$Failed]
            ];
            AppendTo[params,"screen_name"->name]
            ,
            AppendTo[params,"user_id"->getAuthenticatedTwitterID[id]]
        ];
    ];

    expectedSize = FromDigits@params["count"];
    params["count"] = "200";

    While[ size < expectedSize && retries < maxRetries,
            rawdata=OAuthClient`rawoauthdata[id,"RawUserTimeline",Normal@params];
            res=twitterimport[rawdata];
            If[ Length[res] === 0, (*No more tweets available*)
                Break[]
            ];
    
            tmp = Flatten[#["text"]& /@ Flatten[#["entities"]["hashtags"]& /@ res]];
            result = Join[result, Take[tmp,Min[expectedSize - size, Length[tmp]]]];
            size = Length[result];
    
            params = Association@params;
            params["max_id"] = ToString[Last[res]["id"] - 1];
            params = Normal@params;
    
            retries++;
    ];

    result
]

twittercookeddata["UserMentions",id_,args_]:=Module[
    {invalidParameters,newargs=KeyMap[ToString]@Association@args,params=<|"tweet_mode"->"extended"|>,res,user,name,tmp,limit,rawdata,expectedSize=0,retries=0,maxRetries=10,size=0,result={}},

    invalidParameters = Select[Keys[newargs],!MemberQ[{"UserID","Username","MaxItems","Elements"},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Twitter"]&/@invalidParameters;
        Throw[$Failed]
    ];

    If[ KeyExistsQ[newargs, "MaxItems"],
        limit = Lookup[newargs, "MaxItems"];
        If[ !(IntegerQ[limit] && 200>limit>0),
            Message[ServiceExecute::nval,MaxItems,"Twitter"];
            Throw[$Failed]
        ];
        AppendTo[params,"count"->ToString@limit],
        AppendTo[params,"count"->"20"]
    ];

    If[ KeyExistsQ[newargs, "UserID"],
        user = ToString@Lookup[newargs, "UserID"];
        If[ !StringMatchQ[user, DigitCharacter..],
            Message[ServiceExecute::nval,"UserID","Twitter"];
            Throw[$Failed]
        ];
        AppendTo[params,"user_id"->user]
        ,
        If[ KeyExistsQ[newargs, "Username"],
            name = Lookup[newargs, "Username"];
            If[ !StringQ[name],
                Message[ServiceExecute::nval,"Username","Twitter"];
                Throw[$Failed]
            ];
            AppendTo[params,"screen_name"->name]
            ,
            AppendTo[params,"user_id"->getAuthenticatedTwitterID[id]]
        ];
    ];

    expectedSize = FromDigits@params["count"];
    params["count"] = "200";

    While[ size < expectedSize && retries < maxRetries,
           rawdata=OAuthClient`rawoauthdata[id,"RawUserTimeline",Normal@params];
           res=twitterimport[rawdata];
           If[ Length[res] === 0, (*No more tweets available*)
               Break[]
           ];
   
           tmp = Flatten[#["screen_name"]& /@ Flatten[#["entities"]["user_mentions"]& /@ res]];
           result = Join[result, Take[tmp,Min[expectedSize - size, Length[tmp]]]];
           size = Length[result];
   
           params = Association@params;
           params["max_id"] = ToString[Last[res]["id"] - 1];
           params = Normal@params;
   
           retries++;
    ];

    result
]

twittercookeddata["UserReplies",id_,args_]:=Module[
    {invalidParameters,newargs=KeyMap[ToString]@Association@args,params=<|"tweet_mode"->"extended"|>,res,user,name,temp,pos,limit,rawdata,expectedSize=0,retries=0,maxRetries=10,size=0,result={}},

    invalidParameters = Select[Keys[newargs],!MemberQ[{"UserID","Username","MaxItems","Elements"},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Twitter"]&/@invalidParameters;
        Throw[$Failed]
    ];

    If[ KeyExistsQ[newargs, "MaxItems"],
        limit = Lookup[newargs, "MaxItems"];
        If[ !(IntegerQ[limit] && 200>limit>0),
            Message[ServiceExecute::nval,MaxItems,"Twitter"];
            Throw[$Failed]
        ];
        AppendTo[params,"count"->ToString@limit],
        AppendTo[params,"count"->"20"]
    ];

    If[ KeyExistsQ[newargs, "UserID"],
        user = ToString@Lookup[newargs, "UserID"];
        If[ !StringMatchQ[user, DigitCharacter..],
            Message[ServiceExecute::nval,"UserID","Twitter"];
            Throw[$Failed]
        ];
        AppendTo[params,"user_id"->user]
        ,
        If[ KeyExistsQ[newargs, "Username"],
            name = Lookup[newargs, "Username"];
            If[ !StringQ[name],
                Message[ServiceExecute::nval,"Username","Twitter"];
                Throw[$Failed]
            ];
            AppendTo[params,"screen_name"->name]
            ,
            AppendTo[params,"user_id"->getAuthenticatedTwitterID[id]]
        ];
    ];

    expectedSize = FromDigits@params["count"];
    params["count"] = "200";

    While[size < expectedSize && retries < maxRetries,
        rawdata=OAuthClient`rawoauthdata[id,"RawUserTimeline",Normal@params];
        res=twitterimport[rawdata];
        If[ Length[res] === 0, (*No more tweets available*)
            Break[]
        ];

        temp=(Lookup[#,"in_reply_to_user_id_str",Null]&/@res);
        pos=Flatten[Position[temp,Except[Null],{1},Heads->False]];
        temp=(<|"ID"->Lookup[#,"id_str",Null], "Text"->Lookup[#,"full_text",Null]|>&/@res[[pos]]);

        result = Join[result, Take[temp,Min[expectedSize - size, Length[temp]]]];
        size = Length[result];

        params = Association@params;
        params["max_id"] = ToString[Last[res]["id"] - 1];
        params = Normal@params;

        retries++;
    ];

    result
]

twittercookeddata["Tweet",id_,args_]:=Module[
    {invalidParameters,newargs=KeyMap[ToString]@Association@args,params=<|"tweet_mode"->"extended"|>,reply,position,rawdata,images,mediaResponses},
    
    invalidParameters = Select[Keys[newargs],!MemberQ[{"Message","GeoPosition","InReplyToStatusID","Image","Elements"},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Twitter"]&/@invalidParameters;
        Throw[$Failed]
    ];

    If[ KeyExistsQ[newargs,"Message"],
        params["status"] = newargs["Message"],
        params["status"] = ""
    ];
    If[ KeyExistsQ[newargs,"InReplyToStatusID"],
        reply = ToString@newargs["InReplyToStatusID"];
        If[ !StringMatchQ[reply, DigitCharacter..],
            Message[ServiceExecute::nval,"InReplyToStatusID","Twitter"];
            Throw[$Failed]
        ];
        params["in_reply_to_status_id"] = reply;
    ];
    If[ !KeyExistsQ[newargs,"Image"],
        images = {},
        images = Flatten[{newargs["Image"]}];
    ];
    If[ Length[newargs["Image"]] > 4,
        Message[ServiceExecute::maximg,"Twitter"];
        Throw[$Failed]
    ];

    mediaResponses = TwitterUploadImages[id, images];
    If[ Length[mediaResponses] > 0,
        params["media_ids"] = StringJoin[Riffle[#["media_id_string"] & /@ mediaResponses, ","]];
        ,
        If[ StringLength[params["status"]] === 0,
            Message[ServiceExecute::nparam,"Message"];
            Throw[$Failed]
        ];
    ];
    
    If[ KeyExistsQ[newargs, "GeoLocation"],
        position = Lookup[newargs,"GeoLocation"];
        If[ MatchQ[GeoPosition[position], GeoPosition[{_?NumericQ, _?NumericQ}]] || MatchQ[GeoPosition[position], GeoPosition[{_?NumericQ, _?NumericQ, _?NumericQ}]],
            params["lat"] = ToString[QuantityMagnitude@Latitude[position]];
            params["long"] = ToString[QuantityMagnitude@Longitude[position]];
            ,
            Message[ServiceExecute::nval,"GeoPosition","Twitter"];
            Throw[$Failed]
        ];
    ];

    rawdata=OAuthClient`rawoauthdata[id,"RawUpdate",Normal[params]];
    twitterimport[rawdata]["text"]

]

twittercookeddata["RateLimit",id_,args_]:=Module[
    {invalidParameters, rawdata, res},
    invalidParameters = Select[Keys[args],!MemberQ[{},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Twitter"]&/@invalidParameters;
        Throw[$Failed]
    ];

    rawdata=OAuthClient`rawoauthdata[id,"RawAccountStatus"];
    res=twitterimport[rawdata];
    If[ KeyExistsQ[res, "resources"],
        Dataset[Association[Replace[res["resources"], Rule[a_String, b_] :> (StringReplace[a, {StartOfString ~~ "/" -> "", ":" -> "", "/" -> " "}] -> b), {2, 5}]]],
        $Failed
    ]
]

twittercookeddata["UserIDSearch",id_,args_]:=Module[
    {invalidParameters,newargs=KeyMap[ToString]@Association@args,params=<|"tweet_mode"->"extended"|>,limit,query,rawdata,res},

    invalidParameters = Select[Keys[newargs],!MemberQ[{"Query","MaxItems","Elements"},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Twitter"]&/@invalidParameters;
        Throw[$Failed]
    ];

    If[ KeyExistsQ[newargs,"Query"],
        query = newargs["Query"];
        If[ StringQ[query],
            params["q"] = newargs["Query"],
            Message[ServiceExecute::nval,"Query","Twitter"];
            Throw[$Failed]
        ],
        Message[ServiceExecute::nparam,"Query"];
        Throw[$Failed]
    ];

    If[ KeyExistsQ[newargs, "MaxItems"],
        limit = Lookup[newargs, "MaxItems"];
        If[ !(IntegerQ[limit] && 21>limit>0),
            Message[ServiceExecute::nval,MaxItems,"Twitter"];
            Throw[$Failed]
        ];
        AppendTo[params,"count"->ToString@limit],
        AppendTo[params,"count"->"20"]
    ];

    rawdata=OAuthClient`rawoauthdata[id,"RawUserSearch",Normal[params]];
    res=twitterimport[rawdata];
    Lookup[#,"id_str"]&/@res
]

twittercookeddata[prop:"FollowerMentionNetwork"|"FollowerReplyToNetwork"|"FriendMentionNetwork"|"FriendReplyToNetwork"|"FriendNetwork"|"FollowerNetwork",id_,args_]:=Module[
    {invalidParameters,newargs=KeyMap[ToString]@Association@args,params=<||>,user,name,result},

    invalidParameters = Select[Keys[args],!MemberQ[{"UserID","Username"},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Twitter"]&/@invalidParameters;
        Throw[$Failed]
    ];
    If[ KeyExistsQ[newargs, "UserID"],
        user = ToString@Lookup[newargs, "UserID"];
        If[ !StringMatchQ[user, DigitCharacter..],
            Message[ServiceExecute::nval,"UserID","Twitter"];
            Throw[$Failed]
        ];
        AppendTo[params,"UserID"->user]
        ,
        If[ KeyExistsQ[newargs, "Username"],
            name = Lookup[newargs, "Username"];
            If[ !StringQ[name],
                Message[ServiceExecute::nval,"Username","Twitter"];
                Throw[$Failed]
            ];
            AppendTo[params,"Username"->name]
            ,
            AppendTo[params,"UserID"->getAuthenticatedTwitterID[id]]
        ];
    ];

    result = TwitterBuildnetwork[prop,id,Normal@params];
    result
]

twittercookeddata[prop:("SearchNetwork"|"SearchReplyToNetwork"|"SearchMentionNetwork"),id_,args_]:=TwitterBuildnetwork[prop,id,args]

twittercookeddata["Retweet",id_,args_]:=Module[
    {invalidParameters,newargs=KeyMap[ToString]@Association@args,params=<|"tweet_mode"->"extended"|>,tweet,rawdata},

    invalidParameters = Select[Keys[newargs],!MemberQ[{"TweetID"},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Twitter"]&/@invalidParameters;
        Throw[$Failed]
    ];

    If[ KeyExistsQ[newargs,"TweetID"],
    	tweet = ToString@newargs["TweetID"];
        If[ !StringMatchQ[tweet, DigitCharacter..],
            Message[ServiceExecute::nval,"TweetID","Twitter"];
            Throw[$Failed]
        ];
        params["id"] = tweet,
        Message[ServiceExecute::nparam,"TweetID"];
        Throw[$Failed]
    ];

    params["trim_user"] = "false";

    rawdata=OAuthClient`rawoauthdata[id,"RawRetweet",Normal[params]];
    twitterimport[rawdata]["text"]
]


(* Send Message *)
twittersendmessage[id_,message_String]:=twittercookeddata["Tweet",id,{"Message"->message}]
twittersendmessage[id_,message_]:=twittercookeddata["Tweet",id,{"Image"->message}]/;ImageQ[message]||MatchQ[message,(_Graphics|_Graphics3D)]
twittersendmessage[___]:=$Failed


bird=Image[RawArray["Byte", {{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0,
  0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},
  {0, 0, 0, 0}, {40, 170, 225, 9}, {40, 170, 225, 113}, {40, 170, 225, 207}, {40, 170, 225, 249}, {40, 170, 225, 252},
  {40, 170, 225, 217}, {40, 170, 225, 131}, {40, 170, 225, 16}, {0, 0, 0, 0}, {0, 0, 0, 0}, {40, 170, 225, 1}, {40,
  170, 225, 74}, {40, 170, 225, 19}}, {{0, 0, 0, 0}, {40, 170, 225, 62}, {40, 170, 225, 124}, {0, 0, 0, 0}, {0, 0, 0,
  0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},
  {0, 0, 0, 0}, {0, 0, 0, 0}, {40, 170, 225, 28}, {40, 170, 225, 210}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40,
  170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 230}, {40, 170, 225,
  86}, {40, 170, 225, 130}, {40, 170, 225, 212}, {40, 170, 225, 187}, {0, 0, 0, 0}}, {{0, 0, 0, 0}, {40, 170, 225,
  158}, {40, 170, 225, 255}, {40, 170, 225, 119}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0,
  0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {40, 170, 225, 5}, {40, 170, 225, 207},
  {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170,
  225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 228},
  {40, 170, 225, 31}, {40, 170, 225, 33}}, {{0, 0, 0, 0}, {40, 170, 225, 201}, {40, 170, 225, 255}, {40, 170, 225,
  255}, {40, 170, 225, 153}, {40, 170, 225, 8}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},
  {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {40, 170, 225, 102}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170,
  225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255},
  {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 252}, {40, 170, 225, 175}, {40, 170, 225, 206}, {40, 170,
  225, 117}}, {{0, 0, 0, 0}, {40, 170, 225, 201}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40,
  170, 225, 213}, {40, 170, 225, 61}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0,
  0}, {0, 0, 0, 0}, {40, 170, 225, 189}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225,
  255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40,
  170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 139}, {0, 0, 0, 0}}, {{0, 0, 0, 0}, {40,
  170, 225, 153}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225,
  255}, {40, 170, 225, 177}, {40, 170, 225, 60}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},
  {40, 170, 225, 221}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170,
  225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255},
  {40, 170, 225, 255}, {40, 170, 225, 111}, {0, 0, 0, 0}, {0, 0, 0, 0}}, {{0, 0, 0, 0}, {40, 170, 225, 59}, {40, 170,
  225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255},
  {40, 170, 225, 255}, {40, 170, 225, 217}, {40, 170, 225, 138}, {40, 170, 225, 75}, {40, 170, 225, 26}, {40, 170,
  225, 1}, {40, 170, 225, 212}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255},
  {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170,
  225, 255}, {40, 170, 225, 255}, {40, 170, 225, 16}, {0, 0, 0, 0}, {0, 0, 0, 0}}, {{0, 0, 0, 0}, {0, 0, 0, 0}, {40,
  170, 225, 154}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225,
  255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40,
  170, 225, 251}, {40, 170, 225, 252}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225,
  255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40,
  170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 8}, {0, 0, 0, 0}, {0, 0, 0, 0}}, {{0, 0, 0, 0}, {40, 170, 225,
  170}, {40, 170, 225, 120}, {40, 170, 225, 215}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40,
  170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225,
  255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40,
  170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225,
  255}, {40, 170, 225, 255}, {40, 170, 225, 239}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}, {{0, 0, 0, 0}, {40, 170,
  225, 177}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255},
  {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170,
  225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255},
  {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170,
  225, 255}, {40, 170, 225, 255}, {40, 170, 225, 198}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}, {{0, 0, 0, 0}, {40,
  170, 225, 91}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225,
  255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40,
  170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225,
  255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40,
  170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 140}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}, {{0, 0, 0, 0},
  {40, 170, 225, 3}, {40, 170, 225, 197}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170,
  225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255},
  {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170,
  225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255},
  {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 67}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}, {{0, 0, 0,
  0}, {0, 0, 0, 0}, {40, 170, 225, 21}, {40, 170, 225, 201}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225,
  255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40,
  170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225,
  255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40,
  170, 225, 255}, {40, 170, 225, 226}, {40, 170, 225, 3}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}, {{0, 0, 0, 0},
  {0, 0, 0, 0}, {0, 0, 0, 0}, {40, 170, 225, 12}, {40, 170, 225, 104}, {40, 170, 225, 217}, {40, 170, 225, 255}, {40,
  170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225,
  255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40,
  170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225,
  255}, {40, 170, 225, 118}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}, {{0, 0, 0, 0}, {0, 0, 0, 0}, {0,
  0, 0, 0}, {40, 170, 225, 144}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255},
  {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170,
  225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255},
  {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 231}, {40, 170,
  225, 11}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}, {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {40,
  170, 225, 21}, {40, 170, 225, 232}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225,
  255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40,
  170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225,
  255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 89}, {0, 0, 0, 0}, {0, 0, 0, 0},
  {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}, {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {40, 170, 225,
  48}, {40, 170, 225, 230}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40,
  170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225,
  255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40,
  170, 225, 255}, {40, 170, 225, 165}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0,
  0}}, {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {40, 170, 225, 16}, {40, 170, 225, 131},
  {40, 170, 225, 216}, {40, 170, 225, 252}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170,
  225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255},
  {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 201}, {40, 170, 225, 10}, {0, 0, 0,
  0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}, {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0,
  0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {40, 170, 225, 59}, {40, 170, 225, 215}, {40, 170, 225,
  255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40,
  170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225,
  201}, {40, 170, 225, 15}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0,
  0, 0}}, {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {40, 170, 225, 8}, {40, 170, 225, 74}, {40, 170,
  225, 183}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255},
  {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170,
  225, 255}, {40, 170, 225, 255}, {40, 170, 225, 172}, {40, 170, 225, 11}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},
  {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}, {{40, 170, 225, 81}, {40, 170, 225, 147},
  {40, 170, 225, 167}, {40, 170, 225, 195}, {40, 170, 225, 248}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170,
  225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255},
  {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 235}, {40, 170,
  225, 95}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0,
  0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}, {{0, 0, 0, 0}, {40, 170, 225, 62}, {40, 170, 225, 179}, {40, 170, 225, 251},
  {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170,
  225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255}, {40, 170, 225, 255},
  {40, 170, 225, 224}, {40, 170, 225, 120}, {40, 170, 225, 12}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0,
  0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}, {{0, 0, 0,
  0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {40, 170, 225, 22}, {40, 170, 225, 101}, {40, 170, 225, 156}, {40, 170, 225, 211},
  {40, 170, 225, 235}, {40, 170, 225, 253}, {40, 170, 225, 255}, {40, 170, 225, 246}, {40, 170, 225, 224}, {40, 170,
  225, 184}, {40, 170, 225, 130}, {40, 170, 225, 60}, {40, 170, 225, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0,
  0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0,
  0, 0}, {0, 0, 0, 0}}}], "Byte", ColorSpace -> "RGB", Interleaving -> True];


End[] (* End Private Context *)

End[]


SetAttributes[{},{ReadProtected, Protected}];

(* Return three functions to define oauthservicedata, oauthcookeddata, oauthsendmessage  *)
{TwitterOAuth`Private`twitterdata,TwitterOAuth`Private`Twittercookeddata,TwitterOAuth`Private`twittersendmessage}
