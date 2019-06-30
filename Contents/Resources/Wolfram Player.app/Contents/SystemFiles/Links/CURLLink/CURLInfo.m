(* Mathematica Package *)

BeginPackage["CURLInfo`",{"CURLLink`"}]
(* Exported symbols added here with SymbolName::usage *)  

CURLEEnum::usage = "CURLEEnum "
CURLLink`$CURLMultiOptions
CURLLink`CURLMultiOptionQ
CURLInfo`libcurlFeatures
CURLInfo`protocols
Begin["`Private`"] (* Begin Private Context *) 

$CURLPostRedir = 3

CURLLink`$CURLOptions = {
	"CURLOPT_WRITEFUNCTION"	->	20011,
	"CURLOPT_WRITEDATA"	->	10001,
	"CURLOPT_READFUNCTION"	->	20012,
	"CURLOPT_READDATA"	->	10009,
	"CURLOPT_IOCTLFUNCTION"	->	20130,
	"CURLOPT_IOCTLDATA"	->	10131,
	"CURLOPT_SEEKFUNCTION"	->	20167,
	"CURLOPT_SEEKDATA"	->	10168,
	"CURLOPT_SOCKOPTFUNCTION"	->	20148,
	"CURLOPT_SOCKOPTDATA"	->	10149,
	"CURLOPT_OPENSOCKETFUNCTION"	->	20163,
	"CURLOPT_OPENSOCKETDATA"	->	10164,
	"CURLOPT_NOPROGRESS"->43,
	"CURLOPT_VERBOSE"->41,
	"CURLOPT_PROGRESSFUNCTION"	->	20056,
	"CURLOPT_PROGRESSDATA"	->	10057,
	"CURLOPT_XFERINFOFUNCTION"->20219,
	"CURLOPT_XFERINFODATA"->10057,
	"CURLOPT_HEADERFUNCTION"	->	20079,
	"CURLOPT_WRITEHEADER"	->	10029,
	"CURLOPT_HEADERDATA"	->	10029,
	"CURLOPT_DEBUGFUNCTION"	->	20094,
	"CURLOPT_DEBUGDATA"	->	10095,
	"CURLOPT_SSL_CTX_FUNCTION"	->	20108,
	"CURLOPT_SSL_CTX_DATA"	->	10109,
	"CURLOPT_CONV_TO_NETWORK_FUNCTION"	->	20143,
	"CURLOPT_CONV_FROM_NETWORK_FUNCTION"	->	20142,
	"CURLOPT_CONV_FROM_UTF8_FUNCTION"	->	20144,
	"CURLOPT_ERRORBUFFER"	->	10010,
	"CURLOPT_STDERR"	->	10037,
	"CURLOPT_FAILONERROR"	->	45,
	"CURLOPT_URL"	->	10002,
	"CURLOPT_PROTOCOLS"	->	181,
	"CURLOPT_REDIR_PROTOCOLS"	->	182,
	"CURLOPT_PROXY"	->	10004,
	"CURLOPT_PROXYPORT"	->	59,
	"CURLOPT_PROXYTYPE"	->	101,
	"CURLOPT_NOPROXY"	->	10177,
	"CURLOPT_HTTPPROXYTUNNEL"	->	61,
	"CURLOPT_SOCKS5_GSSAPI_SERVICE"	->	10179,
	"CURLOPT_SOCKS5_GSSAPI_NEC"	->	180,
	"CURLOPT_INTERFACE"	->	10062,
	"CURLOPT_LOCALPORT"	->	139,
	"CURLOPT_LOCALPORTRANGE"	->	140,
	"CURLOPT_DNS_CACHE_TIMEOUT"	->	92,
	"CURLOPT_DNS_USE_GLOBAL_CACHE"	->	91,
	"CURLOPT_BUFFERSIZE"	->	98,
	"CURLOPT_PORT"	->	3,
	"CURLOPT_TCP_NODELAY"	->	121,
	"CURLOPT_TCP_KEEPALIVE"->213,
	"CURLOPT_TCP_KEEPIDLE"->214,
	"CURLOPT_TCP_KEEPINTVL"->215,
	"CURLOPT_ADDRESS_SCOPE"	->	171,
	"CURLOPT_NETRC"	->	51,
	"CURLOPT_NETRC_FILE"	->	10118,
	"CURLOPT_USERPWD"	->	10005,
	"CURLOPT_PROXYUSERPWD"	->	10006,
	"CURLOPT_USERNAME"	->	10173,
	"CURLOPT_PASSWORD"	->	10174,
	"CURLOPT_PROXYUSERNAME"	->	10175,
	"CURLOPT_PROXYPASSWORD"	->	10176,
	"CURLOPT_HTTPAUTH"	->	107,
	"CURLOPT_PROXYAUTH"	->	111,
	"CURLOPT_AUTOREFERER"	->	58,
	"CURLOPT_FOLLOWLOCATION"	->	52,
	"CURLOPT_UNRESTRICTED_AUTH"	->	105,
	"CURLOPT_MAXREDIRS"	->	68,
	"CURLOPT_POSTREDIR"	->	161,
	"CURLOPT_PUT"	->	54,
	"CURLOPT_POST"	->	47,
	"CURLOPT_POSTFIELDS"	->	10015,
	"CURLOPT_POSTFIELDSIZE"	->	60,
	"CURLOPT_POSTFIELDSIZE_LARGE"	->	30120,
	"CURLOPT_COPYPOSTFIELDS"	->	10165,
	"CURLOPT_HTTPPOST"	->	10024,
	"CURLOPT_REFERER"	->	10016,
	"CURLOPT_USERAGENT"	->	10018,
	"CURLOPT_HTTPHEADER"	->	10023,
	"CURLOPT_HTTP200ALIASES"	->	10104,
	"CURLOPT_COOKIE"	->	10022,
	"CURLOPT_COOKIEFILE"	->	10031,
	"CURLOPT_COOKIEJAR"	->	10082,
	"CURLOPT_COOKIESESSION"	->	96,
	"CURLOPT_COOKIELIST"	->	10135,
	"CURLOPT_HTTPGET"	->	80,
	"CURLOPT_HTTP_VERSION"	->	84,
	"CURLOPT_IGNORE_CONTENT_LENGTH"	->	136,
	"CURLOPT_HTTP_CONTENT_DECODING"	->	158,
	"CURLOPT_HTTP_TRANSFER_DECODING"	->	157,
	"CURLOPT_TFTP_BLKSIZE"	->	178,
	"CURLOPT_FTPPORT"	->	10017,
	"CURLOPT_QUOTE"	->	10028,
	"CURLOPT_POSTQUOTE"	->	10039,
	"CURLOPT_PREQUOTE"	->	10093,
	"CURLOPT_DIRLISTONLY"	->	48,
	"CURLOPT_APPEND"	->	50,
	"CURLOPT_FTP_USE_EPRT"	->	106,
	"CURLOPT_FTP_USE_EPSV"	->	85,
	"CURLOPT_FTP_CREATE_MISSING_DIRS"	->	110,
	"CURLOPT_FTP_RESPONSE_TIMEOUT"	->	112,
	"CURLOPT_FTP_ALTERNATIVE_TO_USER"	->	10147,
	"CURLOPT_FTP_SKIP_PASV_IP"	->	137,
	"CURLOPT_FTPSSLAUTH"	->	129,
	"CURLOPT_FTP_SSL_CCC"	->	154,
	"CURLOPT_FTP_ACCOUNT"	->	10134,
	"CURLOPT_FTP_FILEMETHOD"	->	138,
	"CURLOPT_TRANSFERTEXT"	->	53,
	"CURLOPT_PROXY_TRANSFER_MODE"	->	166,
	"CURLOPT_CRLF"	->	27,
	"CURLOPT_RANGE"	->	10007,
	"CURLOPT_RESUME_FROM"	->	21,
	"CURLOPT_RESUME_FROM_LARGE"	->	30116,
	"CURLOPT_CUSTOMREQUEST"	->	10036,
	"CURLOPT_FILETIME"	->	69,
	"CURLOPT_NOBODY"	->	44,
	"CURLOPT_INFILESIZE"	->	14,
	"CURLOPT_INFILESIZE_LARGE"	->	30115,
	"CURLOPT_UPLOAD"	->	46,
	"CURLOPT_MAXFILESIZE"	->	114,
	"CURLOPT_MAXFILESIZE_LARGE"	->	30117,
	"CURLOPT_TIMECONDITION"	->	33,
	"CURLOPT_TIMEVALUE"	->	34,
	"CURLOPT_TIMEOUT"	->	13,
	"CURLOPT_TIMEOUT_MS"	->	155,
	"CURLOPT_LOW_SPEED_LIMIT"	->	19,
	"CURLOPT_LOW_SPEED_TIME"	->	20,
	"CURLOPT_MAX_SEND_SPEED_LARGE"	->	30145,
	"CURLOPT_MAX_RECV_SPEED_LARGE"	->	30146,
	"CURLOPT_MAXCONNECTS"	->	71,
	"CURLOPT_CLOSEPOLICY"	->	72,
	"CURLOPT_FRESH_CONNECT"	->	74,
	"CURLOPT_FORBID_REUSE"	->	75,
	"CURLOPT_CONNECTTIMEOUT"	->	78,
	"CURLOPT_CONNECTTIMEOUT_MS"	->	156,
	"CURLOPT_IPRESOLVE"	->	113,
	"CURLOPT_CONNECT_ONLY"	->	141,
	"CURLOPT_USE_SSL"	->	119,
	"CURLOPT_SSLCERT"	->	10025,
	"CURLOPT_SSLCERTTYPE"	->	10086,
	"CURLOPT_SSLKEY"	->	10087,
	"CURLOPT_SSLKEYTYPE"	->	10088,
	"CURLOPT_KEYPASSWD"	->	10026,
	"CURLOPT_SSLENGINE"	->	10089,
	"CURLOPT_SSLENGINE_DEFAULT"	->	90,
	"CURLOPT_SSLVERSION"	->	32,
	"CURLOPT_SSL_VERIFYPEER"	->	64,
	"CURLOPT_CAINFO"	->	10065,
	"CURLOPT_ISSUERCERT"	->	10170,
	"CURLOPT_CAPATH"	->	10097,
	"CURLOPT_CRLFILE"	->	10169,
	"CURLOPT_SSL_VERIFYHOST"	->	81,
	"CURLOPT_CERTINFO"	->	172,
	"CURLOPT_RANDOM_FILE"	->	10076,
	"CURLOPT_EGDSOCKET"	->	10077,
	"CURLOPT_SSL_CIPHER_LIST"	->	10083,
	"CURLOPT_SSL_SESSIONID_CACHE"	->	150,
	"CURLOPT_KRBLEVEL"	->	10063,
	"CURLOPT_SSH_AUTH_TYPES"	->	151,
	"CURLOPT_SSH_HOST_PUBLIC_KEY_MD5"	->	10162,
	"CURLOPT_SSH_PUBLIC_KEYFILE"	->	10152,
	"CURLOPT_SSH_PRIVATE_KEYFILE"	->	10153,
	"CURLOPT_SSH_KNOWNHOSTS"	->	10183,
	"CURLOPT_SSH_KEYFUNCTION"	->	20184,
	"CURLOPT_PRIVATE"	->	10103,
	"CURLOPT_SHARE"	->	10100,
	"CURLOPT_NEW_FILE_PERMS"	->	159,
	"CURLOPT_NEW_DIRECTORY_PERMS"	->	160,
	"CURLOPT_TELNETOPTIONS"	->	10070,
	"CURLOPT_SSL_OPTIONS"	->	216,
	"CURLOPT_ACCEPT_ENCODING"	->	10102,
	"CURLOPT_DEFAULT_PROTOCOL"	->	10238
}
CURLLink`$CURLMultiOptions=
<|
"CURLMOPT_CHUNK_LENGTH_PENALTY_SIZE"->10,
"CURLMOPT_CONTENT_LENGTH_PENALTY_SIZE"->9,
"CURLMOPT_MAXCONNECTS"->6,
"CURLMOPT_MAX_HOST_CONNECTIONS"->7,
"CURLMOPT_MAX_PIPELINE_LENGTH"->8,
"CURLMOPT_MAX_TOTAL_CONNECTIONS"->13,
"CURLMOPT_PIPELINING"->3,
"CURLMOPT_PIPELINING_SERVER_BL"->10012,
"CURLMOPT_PIPELINING_SITE_BL"->10011,
"CURLMOPT_PUSHDATA"->10015,
"CURLMOPT_PUSHFUNCTION"->20014,
"CURLMOPT_SOCKETDATA"->10002,
"CURLMOPT_SOCKETFUNCTION"->20001,
"CURLMOPT_TIMERDATA"->10005,
"CURLMOPT_TIMERFUNCTION"->20004  
|>
CURLLink`$CURLInfo=
<|
"CURLINFO_EFFECTIVE_URL"			-> <|"Value" -> 1048577,"Type" -> "String"|>, 
"CURLINFO_RESPONSE_CODE"			-> <|"Value" -> 2097154,"Type" -> "Integer"|>,
"CURLINFO_TOTAL_TIME" 				-> <|"Value" -> 3145731,"Type" -> "Real"|>, 
"CURLINFO_NAMELOOKUP_TIME" 			-> <|"Value" -> 3145732,"Type" -> "Real"|>, 
"CURLINFO_CONNECT_TIME" 			-> <|"Value" -> 3145733,"Type" -> "Real"|>, 
"CURLINFO_PRETRANSFER_TIME" 		-> <|"Value" -> 3145734,"Type" -> "Real"|>, 
"CURLINFO_SIZE_UPLOAD" 				-> <|"Value" -> 3145735,"Type" -> "Real"|>, 
"CURLINFO_SIZE_DOWNLOAD" 			-> <|"Value" -> 3145736,"Type" -> "Real"|>,
"CURLINFO_SPEED_DOWNLOAD" 			-> <|"Value" -> 3145737,"Type" -> "Real"|>, 
"CURLINFO_SPEED_UPLOAD" 			-> <|"Value" -> 3145738,"Type" -> "Real"|>, 
"CURLINFO_HEADER_SIZE" 				-> <|"Value" -> 2097163,"Type" -> "Integer"|>, 
"CURLINFO_REQUEST_SIZE" 			-> <|"Value" -> 2097164,"Type" -> "Integer"|>, 
"CURLINFO_SSL_VERIFYRESULT" 		-> <|"Value" -> 2097165,"Type" -> "Integer"|>, 
"CURLINFO_FILETIME" 				-> <|"Value" -> 2097166,"Type" -> "Integer"|>, 
"CURLINFO_CONTENT_LENGTH_DOWNLOAD" 	-> <|"Value" -> 3145743,"Type" -> "Real"|>, 
"CURLINFO_CONTENT_LENGTH_UPLOAD" 	-> <|"Value" -> 3145744,"Type" -> "Real"|>, 
"CURLINFO_STARTTRANSFER_TIME" 		-> <|"Value" -> 3145745,"Type" -> "Real"|>, 
"CURLINFO_CONTENT_TYPE" 			-> <|"Value" -> 1048594,"Type" -> "String"|>, 
"CURLINFO_REDIRECT_TIME" 			-> <|"Value" -> 3145747,"Type" -> "Real"|>,
"CURLINFO_REDIRECT_COUNT" 			-> <|"Value" -> 2097172,"Type" -> "Integer"|>, 
"CURLINFO_PRIVATE" 					-> <|"Value" -> 1048597,"Type" -> "String"|>, 
"CURLINFO_HTTP_CONNECTCODE" 		-> <|"Value" -> 2097174,"Type" -> "Integer"|>, 
"CURLINFO_HTTPAUTH_AVAIL" 			-> <|"Value" -> 2097175,"Type" -> "Integer"|>, 
"CURLINFO_PROXYAUTH_AVAIL" 			-> <|"Value" -> 2097176,"Type" -> "Integer"|>, 
"CURLINFO_OS_ERRNO" 				-> <|"Value" -> 2097177,"Type" -> "Integer"|>, 
"CURLINFO_NUM_CONNECTS" 			-> <|"Value" -> 2097178,"Type" -> "Integer"|>, 
"CURLINFO_SSL_ENGINES" 				-> <|"Value" -> 4194331,"Type" -> "SList"|>, 
"CURLINFO_COOKIELIST" 				-> <|"Value" -> 4194332,"Type" -> "SList"|>, 
"CURLINFO_LASTSOCKET" 				-> <|"Value" -> 2097181,"Type" -> "Integer"|>,
"CURLINFO_FTP_ENTRY_PATH" 			-> <|"Value" -> 1048606,"Type" -> "String"|>, 
"CURLINFO_REDIRECT_URL" 			-> <|"Value" -> 1048607,"Type" -> "String"|>, 
"CURLINFO_PRIMARY_IP" 				-> <|"Value" -> 1048608,"Type" -> "String"|>, 
"CURLINFO_APPCONNECT_TIME" 			-> <|"Value" -> 3145761,"Type" -> "Real"|>, 
"CURLINFO_CERTINFO" 				-> <|"Value" -> 4194338,"Type" -> "SList"|>, 
"CURLINFO_CONDITION_UNMET" 			-> <|"Value" -> 2097187,"Type" -> "Integer"|>, 
"CURLINFO_RTSP_SESSION_ID" 			-> <|"Value" -> 1048612,"Type" -> "String"|>, 
"CURLINFO_RTSP_CLIENT_CSEQ" 		-> <|"Value" -> 2097189,"Type" -> "Integer"|>, 
"CURLINFO_RTSP_SERVER_CSEQ" 		-> <|"Value" -> 2097190,"Type" -> "Integer"|>, 
"CURLINFO_RTSP_CSEQ_RECV" 			-> <|"Value" -> 2097191,"Type" -> "Integer"|>, 
"CURLINFO_PRIMARY_PORT" 			-> <|"Value" -> 2097192,"Type" -> "Integer"|>, 
"CURLINFO_LOCAL_IP" 				-> <|"Value" -> 1048617,"Type" -> "String"|>, 
"CURLINFO_LOCAL_PORT" 				-> <|"Value" -> 2097194,"Type" -> "Integer"|>,
"CURLINFO_TLS_SESSION" 				-> <|"Value" -> 4194347,"Type" -> "SList"|>, 
"CURLINFO_ACTIVESOCKET" 			-> <|"Value" -> 5242924,"Type" -> "Socket"|>, 
"CURLINFO_TLS_SSL_PTR" 				-> <|"Value" -> 4194349,"Type" -> "SList"|>|>
Do[
	CURLLink`CURLOptionQ[First@CURLLink`$CURLOptions[[i]]] := True,
	{i, Length[CURLLink`$CURLOptions]}
]
CURLLink`CURLMultiOptionQ[option_String]:=KeyExistsQ[CURLLink`$CURLMultiOptions,option]
CURLLink`CURLOptionQ[_] := False
CURLLink`CURLMultiOptionQ[_] := False

CURLInfo`protocols=<|"HTTP" -> False, "HTTPS" -> False, "FTP" -> False, "FTPS" -> False, 
 "SCP" -> False, "SFTP" -> False, "TELNET" -> False, "LDAP" -> False, 
 "LDAPS" -> False, "DICT" -> False, "FILE" -> False, "TFTP" -> False, 
 "IMAP" -> False, "IMAPS" -> False, "POP3" -> False, "POP3S" -> False,
  "SMTP" -> False, "SMTPS" -> False, "RTSP" -> False, "RTMP" -> False,
  "RTMPT" -> False, "RTMPE" -> False, "RTMPTE" -> False, 
 "RTMPS" -> False, "RTMPTS" -> False, "GOPHER" -> False, 
 "SMB" -> False, "SMBS" -> False|>;

CURLInfo`libcurlFeatures=
{"Supports IPv6", "Supports Kerberos V4 (when using FTP)", "Supports \
Kerberos V5 authentication for FTP, IMAP, POP3, SMTP and SOCKSv5 \
proxy (Added in 7.40.0)", "Supports SSL (HTTPS/FTPS) (Added in \
7.10)", "Supports HTTP deflate using libz (Added in 7.10)", "Supports \
HTTP NTLM (added in 7.10.6)", "Supports HTTP GSS-Negotiate (added in \
7.10.6)", "libcurl was built with debug capabilities (added in \
7.10.6)", "libcurl was built with memory tracking debug capabilities. \
This is mainly of interest for libcurl hackers. (added in 7.19.6)", \
"libcurl was built with support for asynchronous name lookups, which \
allows more exact timeouts (even on Windows) and less blocking when \
using the multi interface. (added in 7.10.7)", "libcurl was built \
with support for SPNEGO authentication (Simple and Protected GSS-API \
Negotiation Mechanism, defined in RFC 2478.) (added in 7.10.8)", \
"libcurl was built with support for large files. (Added in 7.11.1)", \
"libcurl was built with support for IDNA, domain names with \
international letters. (Added in 7.12.0)", "libcurl was built with \
support for SSPI. This is only available on Windows and makes libcurl \
use Windows-provided functions for Kerberos, NTLM, SPNEGO and Digest \
authentication. It also allows libcurl to use the current user \
credentials without the app having to pass them on. (Added in \
7.13.2)", "libcurl was built with support for GSS-API. This makes \
libcurl use provided functions for Kerberos and SPNEGO \
authentication. It also allows libcurl to use the current user \
credentials without the app having to pass them on. (Added in \
7.38.0)", "libcurl was built with support for character conversions, \
as provided by the CURLOPT_CONV_* callbacks. (Added in 7.15.4)", \
"libcurl was built with support for TLS-SRP. (Added in 7.21.4)", \
"libcurl was built with support for NTLM delegation to a winbind \
helper. (Added in 7.22.0)", "libcurl was built with support for \
HTTP2. (Added in 7.33.0)", "libcurl was built with support for Unix \
domain sockets. (Added in 7.40.0)", "libcurl was built with support \
for Mozilla's Public Suffix List. This makes libcurl ignore cookies \
with a domain that's on the list. (Added in 7.47.0)", "libcurl was \
built with support for HTTPS-proxy. (Added in 7.52.0)", "libcurl was \
built with multiple SSL backends. For details, see \
curl_global_sslset. (Added in 7.56.0)", "Supports HTTP Brotli content \
encoding using libbrotlidec (Added in 7.57.0)"}
(***************************************************************)
End[] (* End Private Context *)

EndPackage[]