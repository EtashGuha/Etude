

BeginPackage["ResourceShingleTransmogrify`"]
Unprotect["ResourceShingleTransmogrify`*"]

$ResourceShingleTransmogrifyDirectory = DirectoryName[System`Private`$InputFileName];

Get[FileNameJoin[{$ResourceShingleTransmogrifyDirectory,#}]]&/@{
    "ResourceShingleTransmogrify.m",
    "TransmogrifyUtilities.m"
  }

EndPackage[]
