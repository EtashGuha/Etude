BeginPackage["CloudObject`"]

Begin["`Private`"]

(* URLFetch *)

Unprotect[CloudObject];

CloudObject /: HoldPattern[URLFetch][CloudObject[url_, ___], arg_:"Content", opts:OptionsPattern[]] :=
    fetchURL[getCloud[url], url, arg, opts]

CloudObject /: HoldPattern[URLFetchAsynchronous][CloudObject[url_, ___], arg_, opts:OptionsPattern[]] :=
    fetchURLAsync[getCloud[url], url, arg, opts]

CloudObject /: HoldPattern[URLFetch][CloudObject[url_, ___], args__, opts:OptionsPattern[]] := (
	Message[URLFetch::argb, URLFetch, 1 + Length[{args}],0,1]; $Failed
)

CloudObject /: HoldPattern[URLSave][CloudObject[url_, ___], file:Automatic|_String|_File:Automatic, content:(_String|_List|All):"Content", opts___?OptionQ] :=
    saveURL[getCloud[url], url, file, content, opts]

CloudObject /: HoldPattern[URLSaveAsynchronous][CloudObject[url_, ___], arg_:Automatic,callback_:Identity, opts:OptionsPattern[]] :=
    saveURLAsync[getCloud[url], url, arg,callback, opts]

saveURL[cloud_, url_, file_, content_. options___] /; useUnauthenticatedRequestQ[cloud] := 
	URLSave[url, file, content, options]

saveURL[cloud_, url_, file_, content_, options___] := 
    Block[{$CloudBase = cloud},
    	authenticatedURLSave[url, file, content, options]
    ]

saveURLAsync[cloud_, url_, arg_, callback_, options___] /; useUnauthenticatedRequestQ[cloud] :=
    URLSaveAsynchronous[url, arg, callback, options]

saveURLAsync[cloud_, url_, arg_, callback_, options___] :=
	Block[{$CloudBase = cloud},
    	authenticatedURLSaveAsynchronous[url, arg, callback, options];
	]

Protect[CloudObject];

End[]

EndPackage[]
