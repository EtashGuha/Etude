(* https://www.w3.org/2011/rdfa-context/rdfa-1.1 *)
(* https://www.mediawiki.org/wiki/Wikibase/Indexing/RDF_Dump_Format#Full_list_of_prefixes *)

BeginPackage["GraphStore`RDF`PrefixData`", {"GraphStore`", "GraphStore`RDF`"}];
Begin["`Private`"];

RDFPrefixData[args___] := With[{res = Catch[iRDFPrefixData[args], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


clear[iRDFPrefixData];
iRDFPrefixData[] := Keys[$prefixes];
iRDFPrefixData[prefix_String, localName : Repeated[_String, {0, 1}]] := Replace[
	$prefixes,
	{
		KeyValuePattern[prefix -> namespace_] :> IRI[namespace <> localName],
		_ :> Missing["UnknownPrefix", prefix]
	}
];
iRDFPrefixData[IRI[i_String], "Prefix"] := FirstCase[
	Normal[$prefixes],
	(prefix_ -> namespace_ /; StringStartsQ[i, namespace]) :> prefix,
	Missing["UnknownNamespace"]
];
iRDFPrefixData[IRI[i_String], "LocalName"] := FirstCase[
	Normal[$prefixes],
	(_ -> namespace_ /; StringStartsQ[i, namespace]) :> StringDrop[i, StringLength[namespace]],
	Missing["UnknownNamespace"]
];
iRDFPrefixData[l_List, rest___] := iRDFPrefixData[#, rest] & /@ l;
iRDFPrefixData[ent_, prop_List, rest___] := iRDFPrefixData[ent, #, rest] & /@ prop;

$prefixes = <|
	"as" -> "https://www.w3.org/ns/activitystreams#",
	"bd" -> "http://www.bigdata.com/rdf#",
	"cat" -> "http://www.w3.org/ns/dcat#",
	"cc" -> "http://creativecommons.org/ns#",
	"cnt" -> "http://www.w3.org/2008/content#",
	"ctag" -> "http://commontag.org/ns#",
	"dc" -> "http://purl.org/dc/terms/",
	"dc11" -> "http://purl.org/dc/elements/1.1/",
	"dcat" -> "http://www.w3.org/ns/dcat#",
	"dct" -> "http://purl.org/dc/terms/",
	"dcterms" -> "http://purl.org/dc/terms/",
	"describedby" -> "http://www.w3.org/2007/05/powder-s#describedby",
	"dqv" -> "http://www.w3.org/ns/dqv#",
	"duv" -> "https://www.w3.org/TR/vocab-duv#",
	"earl" -> "http://www.w3.org/ns/earl#",
	"foaf" -> "http://xmlns.com/foaf/0.1/",
	"geo" -> "http://www.opengis.net/ont/geosparql#",
	"gldp" -> "http://www.w3.org/ns/people#",
	"gr" -> "http://purl.org/goodrelations/v1#",
	"grddl" -> "http://www.w3.org/2003/g/data-view#",
	"ht" -> "http://www.w3.org/2006/http#",
	"ical" -> "http://www.w3.org/2002/12/cal/icaltzd#",
	"ldp" -> "http://www.w3.org/ns/ldp#",
	"license" -> "http://www.w3.org/1999/xhtml/vocab#license",
	"ma" -> "http://www.w3.org/ns/ma-ont#",
	"oa" -> "http://www.w3.org/ns/oa#", "og" -> "http://ogp.me/ns#",
	"ontolex" -> "http://www.w3.org/ns/lemon/ontolex#",
	"org" -> "http://www.w3.org/ns/org#",
	"owl" -> "http://www.w3.org/2002/07/owl#",
	"p" -> "http://www.wikidata.org/prop/",
	"pq" -> "http://www.wikidata.org/prop/qualifier/",
	"pqn" -> "http://www.wikidata.org/prop/qualifier/value-normalized/",
	"pqv" -> "http://www.wikidata.org/prop/qualifier/value/",
	"pr" -> "http://www.wikidata.org/prop/reference/",
	"prn" -> "http://www.wikidata.org/prop/reference/value-normalized/",
	"prov" -> "http://www.w3.org/ns/prov#",
	"prv" -> "http://www.wikidata.org/prop/reference/value/",
	"ps" -> "http://www.wikidata.org/prop/statement/",
	"psn" -> "http://www.wikidata.org/prop/statement/value-normalized/",
	"psv" -> "http://www.wikidata.org/prop/statement/value/",
	"ptr" -> "http://www.w3.org/2009/pointers#",
	"qb" -> "http://purl.org/linked-data/cube#",
	"rdf" -> "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
	"rdfa" -> "http://www.w3.org/ns/rdfa#",
	"rdfs" -> "http://www.w3.org/2000/01/rdf-schema#",
	"rev" -> "http://purl.org/stuff/rev#",
	"rif" -> "http://www.w3.org/2007/rif#",
	"role" -> "http://www.w3.org/1999/xhtml/vocab#role",
	"rr" -> "http://www.w3.org/ns/r2rml#",
	"schema" -> "http://schema.org/",
	"sd" -> "http://www.w3.org/ns/sparql-service-description#",
	"sioc" -> "http://rdfs.org/sioc/ns#",
	"skos" -> "http://www.w3.org/2004/02/skos/core#",
	"skosxl" -> "http://www.w3.org/2008/05/skos-xl#",
	"sosa" -> "http://www.w3.org/ns/sosa/",
	"ssn" -> "http://www.w3.org/ns/ssn/",
	"time" -> "http://www.w3.org/2006/time#",
	"v" -> "http://rdf.data-vocabulary.org/#",
	"vcard" -> "http://www.w3.org/2006/vcard/ns#",
	"void" -> "http://rdfs.org/ns/void#",
	"wd" -> "http://www.wikidata.org/entity/",
	"wdata" -> "http://www.wikidata.org/wiki/Special:EntityData/",
	"wdno" -> "http://www.wikidata.org/prop/novalue/",
	"wdr" -> "http://www.w3.org/2007/05/powder#",
	"wdref" -> "http://www.wikidata.org/reference/",
	"wdrs" -> "http://www.w3.org/2007/05/powder-s#",
	"wds" -> "http://www.wikidata.org/entity/statement/",
	"wdt" -> "http://www.wikidata.org/prop/direct/",
	"wdtn" -> "http://www.wikidata.org/prop/direct-normalized/",
	"wdv" -> "http://www.wikidata.org/value/",
	"wikibase" -> "http://wikiba.se/ontology#",
	"xhv" -> "http://www.w3.org/1999/xhtml/vocab#",
	"xml" -> "http://www.w3.org/XML/1998/namespace",
	"xsd" -> "http://www.w3.org/2001/XMLSchema#"
|>;

End[];
EndPackage[];
