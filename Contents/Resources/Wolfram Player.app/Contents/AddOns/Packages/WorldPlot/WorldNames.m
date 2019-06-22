(* ::Package:: *)

(* :Title: World Names *)

(* :Author: John M. Novak *)

(* :Summary:
This package defines symbols for use with the
Miscellaneous`WorldPlot` package at the global level.  The symbols are 
the names of countries and lists of countries.  The defined lists are
World, NorthAmerica, Europe, SouthAmerica, Asia, Oceania, Africa.
*)

(* :Context: WorldPlot` *)

(* :Package Version: 1.1 *)

(* :Copyright: Copyright 1990-2007, Wolfram Research, Inc.*)

(* :History: V 1.0 by John M. Novak, November 1990 
             V 1.1 by Jeff Adams, January 1995.
                   - updated with new country names.
             Collected into single WorldPlot package to fit
                paclet paradigm for version 6.
                - Brian Van Vertloo, January 2007.
*)

(* :Keywords: nations, cartography *)

(* :Source:
	Esselte Map Service AB (Sweden): The Concise EARTHBOOK World
		Atlas, Earthbooks Incorporated, 1990.
	Country names are conventional short form names found in
		the CIA World Factbook 1994
*)

(* :Mathematica Version: 2.0 *)

(* :Limitation: Does not name every country in the world, but only
	those for which there is data in the WorldData.m package. *)

If[Not@ValueQ[World::usage],World::usage =
"The symbol World contains the names of all the countries in the WorldData \
database."];

World = {"Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antarctica",
   "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", "Bahrain", "Bangladesh", "Belarus",
   "Belgium", "Belize", "Benin", "Bermuda", "Bhutan", "Bolivia", "Bosnia and Herzegovina",
   "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Burma", "Burundi", "CAR", "Cabinda",
   "Cambodia", "Cameroon", "Canada", "Chad", "Chile", "China", "Colombia", "Congo",
   "Costa Rica", "Cote d'Ivoire", "Croatia", "Cuba", "Cyprus", "Czech Republic", "Denmark", "Democratic Republic of the Congo", "Djibouti",
   "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia",
   "Ethiopia", "Falkland Islands", "Fiji", "Finland", "France", "French Guiana",
   "Gabon", "The Gambia", "Georgia", "Germany", "Ghana", "Gibraltar", "Greece", "Greenland",
   "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "Hungary",
   "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy",
    "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kuwait", "Kyrgyzstan",
   "Laos", "Latvia", "Lebanon", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", 
   "Macedonia", "Madagascar", "Malawi",
   "Malaysia", "Mali", "Mauritania", "Mexico", "Moldova", "Monaco", "Mongolia", "Morocco",
   "Mozambique", "Namibia", "Nepal", "Netherlands", "New Zealand", "Nicaragua",
   "Niger", "Nigeria", "North Korea", "Norway", "Oman", "Pakistan", "Panama",
   "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal",
   "Puerto Rico", "Qatar", "Romania", "Russia", "Rwanda", "San Marino", "Saudi Arabia",
   "Senegal", "Serbia and Montenegro", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", 
    "Somalia", "South Africa", 
   "Lesotho", "South Korea", "Spain", "Sri Lanka", "Sudan", "Suriname", "Swaziland", "Sweden",
   "Switzerland", "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Togo", "Tunisia",
   "Turkey", "Turkmenistan", "UAE", "United Kingdom", "USA", "Uganda", "Ukraine", "Uruguay", 
   "Uzbekistan", "Venezuela", "Vietnam", "Western Sahara", "Yemen",
   "Zambia", "Zimbabwe"};

(* Note: Lesotho is out of alphabetical 
order so it can be plotted on top of South
Africa. (otherwise, when a polygon, S.A. covers Lesotho) *)

If[Not@ValueQ[Europe::usage],Europe::usage =
"Europe contains the names of all the countries of Europe defined in the \
WorldData database; includes Russia."];

Europe = {"Bosnia and Herzegovina", "Croatia", "Macedonia", "Slovenia",
	"Portugal", "Spain", "France", "Italy", "Switzerland", "Austria",
	"Germany", "Luxembourg", "Liechtenstein", "Andorra", "Belgium", "Netherlands",
	"Denmark", "Poland", "Czech Republic", "Slovakia", "Hungary", "Albania",
	"Serbia and Montenegro", "Romania", "Bulgaria", "Greece", "United Kingdom", "Turkey", "Cyprus",
	"Ireland", "Iceland", "Norway", "Sweden", "Finland", "Monaco", "Gibraltar", "San Marino",
    "Belarus", "Estonia", "Latvia", "Lithuania", "Moldova", "Ukraine", "Russia"};

If[Not@ValueQ[SouthAmerica::usage],SouthAmerica::usage =
"SouthAmerica contains the names of the countries of South America defined in \
the WorldData database."];

SouthAmerica = {"Colombia", "Venezuela", "Guyana", "Suriname",
	"French Guiana", "Ecuador", "Peru", "Bolivia", "Chile", "Paraguay",
	"Argentina", "Uruguay", "Brazil"};

If[Not@ValueQ[NorthAmerica::usage],NorthAmerica::usage =
"NorthAmerica contains the names of the countries of North and Central \
America defined in the WordData database."];

NorthAmerica = {"USA", "Canada", "Mexico", "Greenland", "Bermuda", "Cuba", "Jamaica", "Haiti",
	"Belize", "Dominican Republic", "El Salvador", "Guatemala", "Honduras", "Nicaragua",
	"Costa Rica", "Panama", "Puerto Rico"};

If[Not@ValueQ[Oceania::usage],Oceania::usage =
"Oceania contains the names of the countries of the South Pacific defined in \
the WorldData database."];

Oceania = {"Indonesia", "Papua New Guinea", "Fiji", "Australia", "New Zealand"};

If[Not@ValueQ[Asia::usage],Asia::usage =
"Asia contains the names of the countries of Asia defined in the WorldData \
database; includes the Middle East and Russia."];

Asia = {"China", "Mongolia", "Afghanistan", "Pakistan", "India",
	"Nepal", "Bhutan", "Sri Lanka", "Bangladesh", "Burma", "Thailand",
	"Laos", "Cambodia", "Vietnam", "North Korea", "South Korea",
	"Japan", "Taiwan", "Turkey", "Lebanon", "Syria", "Iraq", "Iran", "Israel", "Jordan", "Kuwait",
	"Saudi Arabia", "Bahrain", "Qatar", "UAE", "Yemen", "Oman", "Philippines",
	"Malaysia", "Singapore", "Brunei", "Indonesia", "Armenia", "Azerbaijan", "Georgia",
    "Kazakhstan", "Kyrgyzstan", "Tajikistan", "Turkmenistan", "Uzbekistan", "Russia"};

If[Not@ValueQ[Africa::usage],Africa::usage =
"Africa contains the names of the countries of Africa defined in the WorldData \
database."];

Africa = {"Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cameroon",
	"CAR", "Chad", "Congo", "Democratic Republic of the Congo", "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", 
    "Ethiopia", "Gabon", "Ghana",
	"Guinea", "Guinea-Bissau", "Cote d'Ivoire", "Kenya", "Liberia", "Libya",
	"Madagascar", "Malawi", "Mali", "Mauritania", "Morocco", "Mozambique", "Namibia", "Niger",
	"Nigeria", "South Africa", "Lesotho", "Rwanda", "Senegal", "Sierra Leone", "Somalia", 
    "Sudan", "Swaziland", "Tanzania", "The Gambia", "Togo", "Tunisia", "Uganda", 
    "Western Sahara", "Zambia", "Zimbabwe"};

If[Not@ValueQ[MiddleEast::usage],MiddleEast::usage =
"MiddleEast contains the names of the countries of the Middle East defined \
in the WorldData database."];

MiddleEast = {"Egypt", "Israel", "Lebanon", "Syria", "Turkey", "Saudi Arabia", "Yemen",
	"Oman", "UAE", "Bahrain", "Kuwait", "Iraq", "Iran", "Jordan", "Qatar"};

If[Not@ValueQ[USStates::usage],USStates::usage =
"USStates contains the names of the 50 United States, defined in the \
USData database for use with WorldPlot. Set the WorldPlot option \
WorldDatabase -> USData to use this list."];

USStates = {"Alabama", "Alaska", "Arizona", "Arkansas", "California", 
	"Colorado", "Connecticut", "Delaware",  
	"Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana",
	"Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
	"Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri",
	"Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", 
	"New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
	"Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
	"South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia",
	"Washington", "West Virginia", "Wisconsin", "Wyoming"};

If[Not@ValueQ[ContiguousUSStates::usage],ContiguousUSStates::usage =
"ContiguousUSStates contains the names of the 48 contiguous United States, \
defined in the USData database for use with WorldPlot. Set the WorldPlot option \
WorldDatabase -> USData to use this list."];

ContiguousUSStates = {"Alabama", "Arizona", "Arkansas", "California", 
	"Colorado", "Connecticut", "Delaware",  
	"Florida", "Georgia", "Idaho", "Illinois", "Indiana",
	"Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
	"Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri",
	"Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", 
	"New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
	"Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
	"South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia",
	"Washington", "West Virginia", "Wisconsin", "Wyoming"};
