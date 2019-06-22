
BeginPackage["LLVMLink`BuildInfo`"]



Begin["`Private`"]

LLVMLink`Internal`$BuildInfo = <|
    "Date" -> "@GIT_DATE@",
    "Branch" -> "@GIT_BRANCH@",
    "CommitSHA" -> "@GIT_SHA1@",
    "CommitMessage" -> "@GIT_COMMIT_MESSAGE@",
    "RefSpec" -> "@GIT_REFSPEC@"
|>

End[]

EndPackage[]
