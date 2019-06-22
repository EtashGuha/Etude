(* Wolfram Language Package *)

BeginPackage["DataResource`"]

Begin["`Private`"] (* Begin Private Context *) 


$ElementNameLengthLimit=200;
ResourceSystemClient`Private`respositoryMetadataSchema[$DataResourceType]:=(
	ResourceSystemClient`Private`respositoryMetadataSchema[$DataResourceType]=
	Association[
		"Content"->(With[{expr = #}, Interpreter[Evaluate, (SameQ[expr, #] &)][#]] &),
		"ContentElementAccessType"->Restricted["String", RegularExpression[".{1,50}"]],
		"DefaultContentElement"->Restricted["String", RegularExpression[".{1,"<>ToString[$ElementNameLengthLimit]<>"}"]],
		"ContentElements"->RepeatingElement[Restricted["String", RegularExpression[".{1,"<>ToString[$ElementNameLengthLimit]<>"}"]]]
	]
	)


ResourceSystemClient`Private`resourceMaxNameLength[$DataResourceType]:=80;

validateParameter[$DataResourceType,cat:("Categories"|"ContentTypes"),l_List]:=l/;Complement[l,
	ResourceSystemClient`Private`resourceSortingProperties["DataResource"][cat]]==={}
	
validateParameter[$DataResourceType,"ContentElementFunctions",as_]:=as/;AssociationQ[as]

validateParameter[$DataResourceType,"InformationElements",as_]:=as/;AssociationQ[as]
validateParameter[$DataResourceType,"ContentElementFunctions",as_]:=as/;AssociationQ[as]

validateParameter[$DataResourceType,"ContentElementLocations",Automatic]=Automatic;
validateParameter[$DataResourceType,"ContentElementLocations",as_Association]:=validateParameter["DataResource","ContentElementLocations",#]&/@as
validateParameter[$DataResourceType,"ContentElementLocations",co:HoldPattern[_CloudObject]]:=(setReviewerPermissions[co];co)


ResourceSystemClient`Private`repositoryResourceURL[$DataResourceType,info_]:=dataRepositoryShingleURL[Lookup[info,"UUID"],info]

dataRepositoryShingleURL[uuid_,info_]:=(dataRepositoryShingleURL[uuid,_]=datarepositoryShingleURL[uuid, info])

datarepositoryShingleURL[uuid_,info_]:=Block[{
	name=Lookup[info,"ShortName", Lookup[info,"UUID"]], 
	base=Lookup[info,"RepositoryLocation"]},
	base=datarepositorydomain[base];
	If[StringQ[base]&&StringQ[name],
		datarepositoryshingleURL[base,name]
		,
		None
	]
]/;marketplacebasedResourceQ[info]

datarepositoryShingleURL[__]:=None

datarepositoryshingleURL[base_,name_]:=URL[URLBuild[{base,"resources",name}]]

ResourceSystemClient`Private`defaultSortingProperties[$DataResourceType]:=(
ResourceSystemClient`Private`defaultSortingProperties["DataResource"]=
Association[
	"Categories" -> {"Agriculture", "Astronomy", "Atmospheric Science", 
   "Chemistry", "Computational Universe", "Computer Systems", 
   "Culture", "Demographics", "Earth Science", "Ecology", "Economics",
    "Education", "Engineering", "Geography", "Geometry Data", 
   "Government", "Graphics", "Health", "Healthcare", "History", 
   "Human Activities", "Images", "Language", "Life Science", 
   "Literature", "Machine Learning", "Manufacturing", "Mathematics", 
   "Medicine", "Meteorology", "Physics", "Politics", "Reference", 
   "Social Media", "Sociology", "Statistics", "Textual Data", 
   "Transportation"}, 
 "ContentTypes" -> {"Audio", "Entity Store", "Geospatial Data", 
   "Graphs", "Image", "Numerical Data", "Text", "Time Series"}
])

End[] (* End Private Context *)

EndPackage[]