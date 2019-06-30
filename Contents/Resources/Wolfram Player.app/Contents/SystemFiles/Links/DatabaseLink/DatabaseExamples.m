(* Author:          Christopher Williamson *)
(* Copyright:       Copyright 2004-2005, Wolfram Research, Inc. *)

BeginPackage[ "DatabaseLink`DatabaseExamples`", "DatabaseLink`"]

DatabaseExamplesBuild::usage = "DatabaseExamplesBuild[] restores examples databases. 
DatabaseExamplesBuild[name] restores a named example database.";

DatabaseExamplesInstall::usage = "DatabaseExamplesInstall[] installs the demo databases.";

DatabaseExamplesUninstall::usage = "DatabaseExamplesUninstall[] uninstalls the demo databases.";

DatabaseExamplesInstall::direrror = 
"The DatabaseResources directory cannot be created. You should check that Mathematica can
write to $UserBaseDirectory and that it does not contain a file called DatabaseResources."

DatabaseExamplesInstall::example = 
"The Examples directory already exists. You do not need to install the demo examples."

DatabaseExamplesInstall::install = 
"Your DatabaseLink installation does not contain an Examples directory."

DatabaseExamplesInstall::end = 
"Installation of the examples for DatabaseLink has completed."

DatabaseExamplesUninstall::end = 
"The examples for DatabaseLink have been uninstalled."

DatabaseExamplesBuild::conn = "DatabaseExamplesBuild encountered an error while restoring `1`.
Perhaps you have not installed the examples."

Begin["`Package`"];


DatabaseExamplesInstall[] :=
	Module[ {dir, ef},
		dir = Directory[];
		SetDirectory[ $UserBaseDirectory] ;
		ef = DatabaseExamplesInstallImpl[];
		SetDirectory[ dir];
		ef
		]



DatabaseExamplesInstallImpl[] :=
	Module[ {},
		If[ FileType[ "DatabaseResources"] === None,
			CreateDirectory[ "DatabaseResources"]] ;
		If[ FileType[ "DatabaseResources"] =!= Directory,
				Message[ DatabaseExamplesInstall::direrror];
				Return[ $Failed]];
		SetDirectory[ ToFileName[ {$UserBaseDirectory}, "DatabaseResources"]];
		If[ FileType[ "Examples"] =!= None,
				Message[ DatabaseExamplesInstall::example];
				Return[ $Failed]];
		If[ FileType[ ToFileName[ {$DatabaseLinkDirectory}, "Examples"]] =!= Directory,
				Message[ DatabaseExamplesInstall::install];
				Return[ $Failed]];
		CopyDirectory[ ToFileName[ {$DatabaseLinkDirectory}, "Examples"], 
					ToFileName[ {Directory[]}, "Examples"]];
		Message[ DatabaseExamplesInstall::end];
	]

DatabaseExamplesUninstall[] := DatabaseExamplesUninstall[$UserBaseDirectory];
DatabaseExamplesUninstall[dir_] :=
  (
    DeleteDirectory[ ToFileName[{dir, "DatabaseResources"}, "Examples"], DeleteContents -> True];
    Message[ DatabaseExamplesUninstall::end];
  )


SetAttributes[ checkName, Listable];

checkName[ "demo"] := "demo"
checkName[ "publisher"] := "publisher"
checkName[ _] := {}

DatabaseExamplesBuild[] :=
	DatabaseExamplesBuild[ {"demo", "publisher"}]

DatabaseExamplesBuild[ nameIn_] :=
	Module[ {name = nameIn},
		If[ !ListQ[ name], name = {name}];
				
		name = Flatten[ checkName[ name]];
		Catch[ Scan[ DatabaseBuildFun, name], 
				_RestoreException, 
				reportError;
				$Failed]
	]



reportError[ _, RestoreException[ db_]]:=
	Message[ DatabaseExamplesBuild::conn, db]


DatabaseBuildFun[ "demo"] :=
  Module[ {conn, tables, data},
    conn = OpenSQLConnection["demo"];
    If[ conn === $Failed, Throw[ 1, RestoreException[ "demo"]]];
    tables = SQLTables[conn, TableType -> "TABLE"];
    Map[ SQLDropTable[conn, #] &, tables]; 
    SQLCreateTable[conn, "SampleTable1",
        {
      SQLColumn["Entry", DataTypeName -> "INTEGER", Nullable -> True],
      SQLColumn["Value", DataTypeName -> "Double", Nullable -> True],
      SQLColumn["Name", DataTypeName -> "VARCHAR(16)", Nullable -> True]}];
        data = {{1, 5.6, "Day1"},
            {2, 5.9, "Day2"},
            {3, 7.2, "Day3"},
            {4, 6.2, "Day4"},
            {5, 6.0, "Day5"}};
        Map[
          SQLInsert[conn,"SampleTable1",
              {"Entry", "Value", "Name"}, #] &, data];
    CloseSQLConnection[ conn];
  ]


DatabaseBuildFun[ "publisher"] :=
  Module[ {conn, tables},

conn = OpenSQLConnection["publisher"];
If[ conn === $Failed, Throw[ 1, RestoreException[ "publisher"]]];

    tables = SQLTables[conn, TableType -> "TABLE"];
    Map[ SQLDropTable[conn, #] &, tables]; 

SQLExecute[conn, "create table roysched     (title_id char(6) not null,lorange int null,hirange int null,royalty float null)"];
SQLExecute[conn, "create table titleauthors (au_id char(11) not null,title_id char(6) not null,au_ord smallint null,royaltyshare float null)"];
SQLExecute[conn, "create table titleditors  (ed_id char(11) not null,title_id char(6) not null,ed_ord smallint null)"];
SQLExecute[conn, "create table titles       (title_id char(6) not null,title varchar(80) not null,type char(12) null,pub_id char(4) null,price float null, advance float null, ytd_sales int null, contract bit not null,notes varchar(200) null,pubdate date null)"];
SQLExecute[conn, "create table editors      (ed_id char(11) not null,ed_lname varchar(40) not null,ed_fname varchar(20) not null,ed_pos varchar(12) null, phone char(12) null,address varchar(40) null,city varchar(20) null,state char(2) null,zip char(5) null)"];
SQLExecute[conn, "create table sales        (sonum int not null,stor_id char(4) not null,ponum varchar(20) not null, sdate date null)"];
SQLExecute[conn, "create table salesdetails (sonum int not null, qty_ordered smallint not null, qty_shipped smallint null, title_id char(6) not null, date_shipped date null)"];
SQLExecute[conn, "create table authors      (au_id char(11) not null, au_lname varchar(40) not null, au_fname varchar(20) not null, phone char(12) null, address varchar(40) null, city varchar(20) null, state char(2) null, zip char(5) null)"];
SQLExecute[conn, "create table publishers   (pub_id char(4) not null,pub_name varchar(40) null,address varchar(40) null,city varchar(20) null,state char(2) null)"];

SQLExecute[conn, "create unique index taind      on titleauthors (au_id, title_id)"];
SQLExecute[conn, "create unique index edind      on editors      (ed_id)"];
SQLExecute[conn, "create        index ednmind    on editors      (ed_lname, ed_fname)"];
SQLExecute[conn, "create unique index teind      on titleditors  (ed_id, title_id)"];
SQLExecute[conn, "create        index rstidind   on roysched     (title_id)"]SQLExecute[conn, "create unique index pubind     on publishers   (pub_id)"];
SQLExecute[conn, "create unique index auidind    on authors      (au_id)"];
SQLExecute[conn, "create        index aunmind    on authors      (au_lname, au_fname)"];
SQLExecute[conn, "create unique index titleidind on titles       (title_id)"];
SQLExecute[conn, "create        index titleind   on titles       (title)"];

SQLExecute[conn, "insert into authors values ('777-95-6235', 'Kinnison', 'Kimball','415 555-6543', '6223 Drake St.', 'Berkeley', 'CA', '94705')"];
SQLExecute[conn, "insert into authors values ('777-98-5047', 'Garrison', 'Mary','415 555-7543', '309 15th St. #411', 'Oakland', 'CA', '94618')"];
SQLExecute[conn, "insert into authors values ('777-76-7291', 'Cohen', 'Carla','415 555-8675', '589 Darwin Ln.', 'Berkeley', 'CA', '94705')"];
SQLExecute[conn, "insert into authors values ('777-25-9155', 'Costigan', 'Conway','801 555-8653', '67 Twelfth Av.', 'Salt Lake City', 'UT', '84152')"];
SQLExecute[conn, "insert into authors values ('777-35-6309', 'Costigan', 'Clio','801 555-6539', '67 Twelfth Av.', 'Salt Lake City', 'UT', '84152')"];
SQLExecute[conn, "insert into authors values ('777-48-5907', 'Maitland', 'Clifford','219 555-5434', '3 Pate Pl.', 'Gary', 'IN', '46403')"];
SQLExecute[conn, "insert into authors values ('777-36-6585', 'Port', 'Stephanie','301 555-7864', '1956 Armory Pl.', 'Rockville', 'MD', '20853')"];
SQLExecute[conn, "insert into authors values ('777-82-2221', 'McBane', 'Heather','707 555-4504', '301 Pickle', 'Vacaville', 'CA', '95688')"];
SQLExecute[conn, "insert into authors values ('777-20-4923', 'Casity', 'Richard','415 555-0764', '5420 Telly Av.', 'Oakland', 'CA', '94609')"];
SQLExecute[conn, "insert into authors values ('777-35-8737', 'Baker', 'Dirk','415 555-5489', '5420 Tamarack Av.', 'Oakland', 'CA', '94609')"];
SQLExecute[conn, "insert into authors values ('777-94-2490', 'Zwilnik', 'Lydia','415 555-8741', '5720 Culkin St.', 'Oakland', 'CA', '94609')"];
SQLExecute[conn, "insert into authors values ('777-74-8164', 'MacDougall', 'Clarrissa','415 555-4378', '44 Green Terrace Rd.', 'Oakland', 'CA', '94612')"];
SQLExecute[conn, "insert into authors values ('777-42-2040', 'Bright', 'Annabelle','415 555-2463', '3410 Green St.', 'Palo Alto', 'CA', '94301')"];
SQLExecute[conn, "insert into authors values ('777-00-3062', 'Yamazaki', 'Shinya','415 555-8513', '3 Gold Ct.', 'Walnut Creek', 'CA', '94595')"];
SQLExecute[conn, "insert into authors values ('777-47-0758', 'O''Mally', 'Patrick','408 555-1246', '22 Pittsburgh Av. #14', 'San Jose', 'CA', '95128')"];
SQLExecute[conn, "insert into authors values ('777-50-6283', 'Gringle', 'Bart','707 555-1861', 'PO Box 877', 'Covelo', 'CA', '95428')"];
SQLExecute[conn, "insert into authors values ('777-11-5008', 'Porta', 'Mark','615 555-7862', '22 School Rd.', 'Nashville', 'TN', '37215')"];
SQLExecute[conn, "insert into authors values ('777-83-5741', 'VanBuskirk', 'Peter','408 555-1556', '10932 Bulden Rd.', 'Menlo Park', 'CA', '94025')"];
SQLExecute[conn, "insert into authors values ('777-55-7985', 'del Velentia', 'Worsel','615 555-4561', '2286 Graham Pl. #86', 'Ann Arbor', 'MI', '48105')"];
SQLExecute[conn, "insert into authors values ('777-75-0161', 'Holden', 'Marie','415 555-4781', '3410 Green St.', 'Palo Alto', 'CA', '94301')"];
SQLExecute[conn, "insert into authors values ('777-27-5848', 'Lane', 'Charity','415 555-1568', '18 Brook Av.', 'San Francisco', 'CA', '94130')"];
SQLExecute[conn, "insert into authors values ('777-14-2678', 'Barden-Hull', 'Roger','503 555-7865', '55 Scottsdale Bl.', 'Corvallis', 'OR', '97330')"];
SQLExecute[conn, "insert into authors values ('777-43-2522', 'Smithton', 'Meadow','913 555-0156', '10 Plinko Dr.', 'Lawrence', 'KS', '66044')"];

SQLExecute[conn, "insert into publishers values('0736', 'Second Galaxy Books', '100 1st St.','Boston', 'MA')"];
SQLExecute[conn, "insert into publishers values('0877', 'Boskone & Helmuth','201 2nd Ave.', 'Washington', 'DC')"];
SQLExecute[conn, "insert into publishers values('1389', 'NanoSoft Book Publishers', '302 3rd Dr.','Berkeley', 'CA')"];

SQLExecute[conn, "insert into roysched values('BS1011', 0, 5000, .10)"];
SQLExecute[conn, "insert into roysched values('BS1011', 5001, 50000, .12)"];
SQLExecute[conn, "insert into roysched values('CP5018', 0, 2000, .10)"];
SQLExecute[conn, "insert into roysched values('CP5018', 2001, 4000, .12)"];
SQLExecute[conn, "insert into roysched values('CP5018', 4001, 50000, .16)"];
SQLExecute[conn, "insert into roysched values('BS1001', 0, 1000, .10)"];
SQLExecute[conn, "insert into roysched values('BS1001', 1001, 5000, .12)"];
SQLExecute[conn, "insert into roysched values('BS1001', 5001, 7000, .16)"];
SQLExecute[conn, "insert into roysched values('BS1001', 7001, 50000, .18)"];
SQLExecute[conn, "insert into roysched values('PS9999', 0, 50000, .10)"];
SQLExecute[conn, "insert into roysched values('PY2002', 0, 1000, .10)"];
SQLExecute[conn, "insert into roysched values('PY2002', 1001, 5000, .12)"];
SQLExecute[conn, "insert into roysched values('PY2002', 5001, 50000, .14)"];
SQLExecute[conn, "insert into roysched values('PY2003', 0, 2000, .10)"];
SQLExecute[conn, "insert into roysched values('PY2003', 2001, 5000, .12)"];
SQLExecute[conn, "insert into roysched values('PY2003', 5001, 50000, .14)"];
SQLExecute[conn, "insert into roysched values('UK3004', 0, 1000, .10)"];
SQLExecute[conn, "insert into roysched values('UK3004', 1001, 2000, .12)"];
SQLExecute[conn, "insert into roysched values('UK3004', 2001, 6000, .14)"];
SQLExecute[conn, "insert into roysched values('UK3004', 6001, 8000, .18)"];
SQLExecute[conn, "insert into roysched values('UK3004', 8001, 50000, .20)"];
SQLExecute[conn, "insert into roysched values('CK4005', 0, 2000, .10)"];
SQLExecute[conn, "insert into roysched values('CK4005', 2001, 6000, .12)"];
SQLExecute[conn, "insert into roysched values('CK4005', 6001, 8000, .16)"];
SQLExecute[conn, "insert into roysched values('CK4005', 8001, 50000, .16)"];
SQLExecute[conn, "insert into roysched values('CP5010', 0, 5000, .10)"];
SQLExecute[conn, "insert into roysched values('CP5010', 5001, 50000, .12)"];
SQLExecute[conn, "insert into roysched values('PY2012', 0, 5000, .10)"];
SQLExecute[conn, "insert into roysched values('PY2012', 5001, 50000, .12)"];
SQLExecute[conn, "insert into roysched values('PY2013', 0, 5000, .10)"];
SQLExecute[conn, "insert into roysched values('PY2013', 5001, 50000, .12)"];
SQLExecute[conn, "insert into roysched values('UK3006', 0, 1000, .10)"];
SQLExecute[conn, "insert into roysched values('UK3006',1001, 2000, .12)"];
SQLExecute[conn, "insert into roysched values('UK3006', 2001, 6000, .14)"];
SQLExecute[conn, "insert into roysched values('UK3006', 6001, 8000, .18)"];
SQLExecute[conn, "insert into roysched values('UK3006', 8001, 50000, .20)"];
SQLExecute[conn, "insert into roysched values('BS1014', 0, 4000, .10)"];
SQLExecute[conn, "insert into roysched values('BS1014', 4001, 8000, .12)"];
SQLExecute[conn, "insert into roysched values('BS1014', 8001, 50000, .14)"];
SQLExecute[conn, "insert into roysched values('UK3015', 0, 2000, .10)"];
SQLExecute[conn, "insert into roysched values('UK3015', 2001, 4000, .12)"];
SQLExecute[conn, "insert into roysched values('UK3015', 4001, 8000, .14)"];
SQLExecute[conn, "insert into roysched values('UK3015', 8001, 12000, .16)"];
SQLExecute[conn, "insert into roysched values('CK4016', 0, 5000, .10)"];
SQLExecute[conn, "insert into roysched values('CK4016', 5001, 15000, .12)"];
SQLExecute[conn, "insert into roysched values('CK4017', 0, 2000, .10)"];
SQLExecute[conn, "insert into roysched values('CK4017', 2001, 8000, .12)"];
SQLExecute[conn, "insert into roysched values('CK4017', 8001,16000, .14)"];
SQLExecute[conn, "insert into roysched values('BS1007', 0, 5000, .10)"];
SQLExecute[conn, "insert into roysched values('BS1007', 5001,50000, .12)"];
SQLExecute[conn, "insert into roysched values('PY2008', 0, 50000, .10)"];

SQLExecute[conn, "insert into titleauthors values('777-95-6235', 'BS1011', 1, .60)"];
SQLExecute[conn, "insert into titleauthors values('777-27-5848', 'PY2012', 1, 1.00)"];
SQLExecute[conn, "insert into titleauthors values('777-27-5848', 'CP5009', 1, 1.00)"];
SQLExecute[conn, "insert into titleauthors values('777-55-7985', 'UK3015', 1, 1.00)"];
SQLExecute[conn, "insert into titleauthors values('777-83-5741', 'PY2013', 1, 1.00)"];
SQLExecute[conn, "insert into titleauthors values('777-98-5047', 'BS1011', 2, .40)"];
SQLExecute[conn, "insert into titleauthors values('777-76-7291', 'CP5018', 1, 1.00)"];
SQLExecute[conn, "insert into titleauthors values('777-98-5047', 'BS1001', 1, 1.00)"];
SQLExecute[conn, "insert into titleauthors values('777-25-9155', 'PY2002', 1, .50)"];
SQLExecute[conn, "insert into titleauthors values('777-35-6309', 'PY2002', 2, .50)"];
SQLExecute[conn, "insert into titleauthors values('777-25-9155', 'PY2003', 1, 1.00)"];
SQLExecute[conn, "insert into titleauthors values('777-48-5907', 'UK3004', 1, .75)"];
SQLExecute[conn, "insert into titleauthors values('777-35-6309', 'UK3004', 2, .25)"];
SQLExecute[conn, "insert into titleauthors values('777-36-6585', 'CK4005', 1, 1.00)"];
SQLExecute[conn, "insert into titleauthors values('777-35-8737', 'BS1007', 1, 1.00)"];
SQLExecute[conn, "insert into titleauthors values('777-42-2040', 'CP5010', 1, .50)"];
SQLExecute[conn, "insert into titleauthors values('777-75-0161', 'CP5010', 2, .50)"];
SQLExecute[conn, "insert into titleauthors values('777-94-2490', 'PY2008', 1, .75)"];
SQLExecute[conn, "insert into titleauthors values('777-74-8164', 'PY2008', 2, .25)"];
SQLExecute[conn, "insert into titleauthors values('777-74-8164', 'BS1014', 1, .60)"];
SQLExecute[conn, "insert into titleauthors values('777-47-0758', 'BS1014', 2, .40)"];
SQLExecute[conn, "insert into titleauthors values('777-00-3062', 'CK4016', 1, .40)"];
SQLExecute[conn, "insert into titleauthors values('777-47-0758', 'CK4016', 2, .30)"];
SQLExecute[conn, "insert into titleauthors values('777-50-6283', 'CK4016', 3, .30)"];
SQLExecute[conn, "insert into titleauthors values('777-14-2678', 'CK4017', 1, 1.00)"];

SQLExecute[conn, "insert into titles values ('BS1001', 'Designer Class Action Suits','business', '0736', 2.99, 10125.00, 18722, 1,'How to dress for success! This book details the current trends in work fashions.','1985-06-30')"];
SQLExecute[conn, "insert into titles values ('PY2002', 'Self Hypnosis: A Beginner''s Guide','psychology', '0736', 10.95, 2275.00, 2045, 1,'Hypnotise yourself in a snap.  Get yourself out by snapping a second time.','1985-06-15')"];
SQLExecute[conn, "insert into titles values ('PY2003', 'Phobic Psychology','psychology', '0736', 7.00, 6000.00, 111, 1,'An historical and referential guide to phobias.  Includes information on support groups around the U.S. and how to get the most out of them.','1985-10-05')"];
SQLExecute[conn, "insert into titles values ('UK3004', 'Hamburger Again!','mod_cook', '0877', 2.99, 15000.00,22246, 1,'How to turn hamburger into great meals.  Includes the infamous Cheesburger Pie recipe.','1985-06-18')"];
SQLExecute[conn, "insert into titles values ('CK4005', 'Made to Wonder: Cooking the Macabre','trad_cook', '0877', 20.95, 7000.00, 375, 1,'This book is about weird foods from around the world.  Not for the faint of heart.','1985-10-21')"];
SQLExecute[conn, "insert into titles        (title_id, title, pub_id, contract)values('UK3006', 'How to Burn a Compact Disk', '0877', 0)"];
SQLExecute[conn, "insert into titles values ('BS1007', 'Modems for Morons','business', '1389', 19.99, 5000.00, 4095, 1,'Modems made simple.','1985-06-22')"];
SQLExecute[conn, "insert into titles values ('PY2008', 'How Green Is My Valley?','psychology', '0736', 21.59, 7000.00, 375, 1,'How different species perceive color differentiations and how it effects their behavior.','1985-10-21')"];
SQLExecute[conn, "insert into titles        (title_id, title, type, pub_id, contract, notes)values('CP5009', 'The Net: Feeding Trolls and Eating Spam', 'popular_comp', '1389', 0,'Avoid the common pitfalls of internet users through anecdotal wisdom.')"];
SQLExecute[conn, "insert into titles values ('CP5010', 'Taiwan Trails','popular_comp', '1389', 20.00, 8000.00, 4095,1, 'The history of Taiwan with relation to the computer industry.','1985-06-12')"];
SQLExecute[conn, "insert into titles values ('BS1011', 'Guide to Impractical Databases','business', '1389', 19.99, 5000.00, 4095, 1, 'How to avoid the common errors made by new users of database systems.','1985-06-12')"];
SQLExecute[conn, "insert into titles values ('PY2012', 'Know Thyself','psychology', '0736', 7.99, 4000.00, 3336, 1, 'The handbook for the human mind.','1985-06-12')"];
SQLExecute[conn, "insert into titles values ('PY2013', 'Where Minds Meat: The Impact of Diet on Behavior','psychology', '0736', 19.99, 2000.00, 4072,1,'How to change behavior through diet.  Illustrated.','1985-06-12')"];
SQLExecute[conn, "insert into titles values ('BS1014', 'Exit Interviews','business', '1389', 11.95, 5000.00, 3876, 1, 'How to get the most out of future employees by interviewing past employees.', '1985-06-09')"];
SQLExecute[conn, "insert into titles values ('UK3015', 'Treasures of the Sierra Madre','mod_cook', '0877', 19.99, 0.00, 2032, 1, 'Traditional Mexican Cuisine.','1985-06-09')"];
SQLExecute[conn, "insert into titles values ('CK4016', 'Too Many Cooks','trad_cook', '0877', 14.99, 8000.00, 4095, 1, 'Thousands of chefs from around the world share their secret recipes. ','1985-06-12')"];
SQLExecute[conn, "insert into titles values ('CK4017', 'Let Them Eat Cake!','trad_cook', '0877', 11.95, 4000.00, 15096, 1, 'Making pastries can be difficult and time consuming.  This book shows you the secrets to making greate cakes and cookies with a minimum of effort.','1985-06-12')"];
SQLExecute[conn, "insert into titles values ('CP5018', 'Sticky Software: UI and GUI','popular_comp', '1389', 22.95, 7000.00, 8780, 1, 'How to make User Interafaces (UI) and Graphical User Interfaces (GUI) that are user friendly.','1985-06-30')"];

SQLExecute[conn, "insert into editors values ('777-21-9917', 'Kotchanski', 'Kristine', 'project','415 555-7653', '3000 6th St.', 'Berkeley', 'CA', '94710')"];
SQLExecute[conn, "insert into editors values ('777-62-6103', 'Lister', 'David', 'copy','303 555-9873', '15 Sail', 'Denver', 'CO', '80237')"];
SQLExecute[conn, "insert into editors values ('777-05-1527', 'Rimmer', 'Arnold', 'project','415 555-7347', '27 Yosemite', 'Oakland', 'CA', '94609')"];
SQLExecute[conn, "insert into editors values ('777-03-8499', 'Dibley', 'Dwayne', 'copy','312 555-6543', '1010 E. Devon', 'Chicago', 'IL', '60018')"];
SQLExecute[conn, "insert into editors values ('777-78-7915', 'Himmel', 'Eleanore', 'project','617 555-0987', '97 Bleaker', 'Boston', 'MA', '02210')"];
SQLExecute[conn, "insert into editors values ('777-68-5219', 'Rutherford-Hayes', 'Hannah', 'project','301 555-2479', '32 Rockbill Pike', 'Rockbill', 'MD', '20852')"];
SQLExecute[conn, "insert into editors values ('777-53-4715', 'McCann', 'Dennis', 'acquisition','301 555-0783', '32 Rockbill Pike', 'Rockbill', 'MD', '20852')"];
SQLExecute[conn, "insert into editors values ('777-99-6673', 'Kaspchek', 'Christof', 'acquisition','415 555-0064', '18 Severe Rd.', 'Berkeley', 'CA', '94710')"];
SQLExecute[conn, "insert into editors values ('777-88-7514', 'Hunter', 'Amanda', 'acquisition','617 555-6453', '18 Dowdy Ln.', 'Boston', 'MA', '02210')"];

SQLExecute[conn, "insert into titleditors values('777-78-7915', 'BS1001', 2)"];
SQLExecute[conn, "insert into titleditors values('777-78-7915', 'PY2002', 2)"];
SQLExecute[conn, "insert into titleditors values('777-78-7915', 'PY2003', 2)"];
SQLExecute[conn, "insert into titleditors values('777-78-7915', 'PY2013', 2)"];
SQLExecute[conn, "insert into titleditors values('777-78-7915', 'PY2012', 2)"];
SQLExecute[conn, "insert into titleditors values('777-78-7915', 'PY2008', 2)"];
SQLExecute[conn, "insert into titleditors values('777-68-5219', 'UK3015', 2)"];
SQLExecute[conn, "insert into titleditors values('777-68-5219', 'UK3004', 2)"];
SQLExecute[conn, "insert into titleditors values('777-68-5219', 'TC3281', 2)"];
SQLExecute[conn, "insert into titleditors values('777-68-5219', 'CK4017', 2)"];
SQLExecute[conn, "insert into titleditors values('777-68-5219', 'CK4016', 2)"];
SQLExecute[conn, "insert into titleditors values('777-21-9917', 'BS1011', 2)"];
SQLExecute[conn, "insert into titleditors values('777-21-9917', 'BS1014', 2)"];
SQLExecute[conn, "insert into titleditors values('777-21-9917', 'BS1007', 2)"];
SQLExecute[conn, "insert into titleditors values('777-21-9917', 'CP5018', 2)"];
SQLExecute[conn, "insert into titleditors values('777-21-9917', 'CP5010', 2)"];
SQLExecute[conn, "insert into titleditors values('777-21-9917', 'BS1001', 3)"];
SQLExecute[conn, "insert into titleditors values('777-05-1527', 'CP5018', 3)"];
SQLExecute[conn, "insert into titleditors values('777-05-1527', 'CP5010', 3)"];
SQLExecute[conn, "insert into titleditors values('777-99-6673', 'BS1011', 1)"];
SQLExecute[conn, "insert into titleditors values('777-99-6673', 'BS1014', 1)"];
SQLExecute[conn, "insert into titleditors values('777-99-6673', 'BS1001', 1)"];
SQLExecute[conn, "insert into titleditors values('777-99-6673', 'BS1007', 1)"];
SQLExecute[conn, "insert into titleditors values('777-99-6673', 'CP5018', 1)"];
SQLExecute[conn, "insert into titleditors values('777-99-6673', 'CP5010', 1)"];
SQLExecute[conn, "insert into titleditors values('777-53-4715', 'PY2008', 1)"];
SQLExecute[conn, "insert into titleditors values('777-53-4715', 'PY2002', 1)"];
SQLExecute[conn, "insert into titleditors values('777-53-4715', 'PY2003', 1)"];
SQLExecute[conn, "insert into titleditors values('777-53-4715', 'PY2013', 1)"];
SQLExecute[conn, "insert into titleditors values('777-53-4715', 'PY2012', 1)"];
SQLExecute[conn, "insert into titleditors values('777-53-4715', 'UK3015', 1)"];
SQLExecute[conn, "insert into titleditors values('777-53-4715', 'UK3004', 1)"];
SQLExecute[conn, "insert into titleditors values('777-53-4715', 'CK4005', 1)"];
SQLExecute[conn, "insert into titleditors values('777-53-4715', 'CK4017', 1)"];
SQLExecute[conn, "insert into titleditors values('777-53-4715', 'CK4016', 1)"];

SQLExecute[conn, "insert into sales values(1,'7066', 'QA7442.3', '1997-09-13')"];
SQLExecute[conn, "insert into sales values(2,'7067', 'D4482', '1997-09-14')"];
SQLExecute[conn, "insert into sales values(3,'7131', 'N914008', '1997-09-14')"];
SQLExecute[conn, "insert into sales values(4,'7131', 'N914014', '1997-09-14')"];
SQLExecute[conn, "insert into sales values(5,'8042', '423LL922', '1997-09-14')"];
SQLExecute[conn, "insert into sales values(6,'8042', '423LL930', '1997-09-14')"];
SQLExecute[conn, "insert into sales values(7, '6380', '722a', '1997-09-13')"];
SQLExecute[conn, "insert into sales values(8,'6380', '6871', '1997-09-14')"];
SQLExecute[conn, "insert into sales values(9,'8042','P723', '1999-03-11')"];
SQLExecute[conn, "insert into sales values(19,'7896','X999', '1999-02-21')"];
SQLExecute[conn, "insert into sales values(10,'7896','QQ2299', '1999-10-28')"];
SQLExecute[conn, "insert into sales values(11,'7896','TQ456', '1997-12-12')"];
SQLExecute[conn, "insert into sales values(12,'8042','QA879.1', '1997-05-22')"];
SQLExecute[conn, "insert into sales values(13,'7066','A2976', '1997-05-24')"];
SQLExecute[conn, "insert into sales values(14,'7131','P3087a', '1997-05-29')"];
SQLExecute[conn, "insert into sales values(15,'7067','P2121', '1997-06-15')"];

SQLExecute[conn, "insert into salesdetails values(1, 75, 75,'PY2002', '1997-09-15')"];
SQLExecute[conn, "insert into salesdetails values(2, 10, 10,'PY2002', '1997-09-15')"];
SQLExecute[conn, "insert into salesdetails values(3, 20, 720,'PY2002', '1997-09-18')"];
SQLExecute[conn, "insert into salesdetails values(4, 25, 20,'UK3004', '1997-09-18')"];
SQLExecute[conn, "insert into salesdetails values(5, 15, 15,'UK3004', '1997-09-14')"];
SQLExecute[conn, "insert into salesdetails values(6, 10, 3,'BS1011', '1997-09-22')"];
SQLExecute[conn, "insert into salesdetails values(7, 3, 3,'PY2002', '1997-09-20')"];
SQLExecute[conn, "insert into salesdetails values(8, 5, 5,'BS1011', '1997-09-14')"];
SQLExecute[conn, "insert into salesdetails values(9, 25, 5,'BS1014', '1999-03-28')"];
SQLExecute[conn, "insert into salesdetails values(19, 35, 35,'BS1001', '1999-03-15')"];
SQLExecute[conn, "insert into salesdetails values(10, 15, 15,'BS1007', '1999-10-29')"];
SQLExecute[conn, "insert into salesdetails values(11, 10, 10,'UK3015', '1999-01-12')"];
SQLExecute[conn, "insert into salesdetails values(12, 30, 30,'CP5018', '1999-05-24')"];
SQLExecute[conn, "insert into salesdetails values(13, 50, 50,'CP5010', '1999-05-24')"];
SQLExecute[conn, "insert into salesdetails values(14, 20, 20,'PY2008', '1999-05-29')"];
SQLExecute[conn, "insert into salesdetails values(14, 25, 25,'PY2003', '1999-04-29')"];
SQLExecute[conn, "insert into salesdetails values(14, 15, 10,'PY2013', '1999-05-29')"];
SQLExecute[conn, "insert into salesdetails values(14, 25, 25,'PY2012', '1999-06-13')"];
SQLExecute[conn, "insert into salesdetails values(15, 40, 40,'CK4005', '1999-06-15')"];
SQLExecute[conn, "insert into salesdetails values(15, 20, 20,'CK4017', '1999-05-30')"];
SQLExecute[conn, "insert into salesdetails values(15, 20, 10,'CK4016', '1999-06-17')"];
CloseSQLConnection[ conn];
  ]


End[]; (* DatabaseLink`DatabaseExamples`Package` *)


EndPackage[] (* DatabaseLink`DatabaseExamples` *)
