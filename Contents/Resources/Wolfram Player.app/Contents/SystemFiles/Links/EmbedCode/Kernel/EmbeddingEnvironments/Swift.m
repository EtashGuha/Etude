(* Wolfram Language package *)

(* EmbedCode for Data Drop *)
EmbedCode`Common`iEmbedCode["swift", databin_Databin, url_, opts:OptionsPattern[]] :=
	Module[
		{code},
		code = 
        TemplateApply[
        	StringJoin[
        		swiftDataDropUsage <> "\n\n" <>
        		swiftDataDropImport <> "\n\n" <>
        		swiftClassHeader <> "\n\n" <>
	            swiftDataDropRecentFunction <> "\n\n" <>
	            swiftDataDropAddFunction <> "\n\n" <>
	            swiftAuxiliarFunctions  <> "\n\n" <>
	            swiftFooter
	        ],
			Association[
				"binId" -> databin[[1]]
			]
        ];
        
		Association[{
            "EnvironmentName" -> "swift",
            "CodeSection" -> Association["Content" -> code, 
                             (* Leave out Description field to indicate that there should be no description text (redundant when no lib files are present) *)
                              "Description" -> "", 
                             "Title" -> Automatic]
        }]
	];

swiftDataDropUsage= 
"/*
Usage:

var bin = WolframDatabin(idDatabin: \"`binId`\")

//Add
var newData : [String: String] = [\"x\":\"10\", \"y\":\"10\", \"z\":\"30\"]
var addJSON : JSON = [:]
        bin.addData(newData, completionHandler: {
            jsonObj in
            addJSON = jsonObj
            println(\"Add Message\")
            println(addJSON[\"Message\"])
            println(addJSON[\"Data\"])

        })
        
//Recent
var recentJSON : JSON = [:]
        bin.getRecent( {
            jsonObj in
            recentJSON = jsonObj
            var record : JSON = recentJSON[0]
            println(\"Data\")
            println(record[\"Data\"])
        })
        
*/
"

swiftDataDropImport = "
import UIKit
import Alamofire
import SwiftyJSON
";

swiftClassHeader = "
class WolframDatabin {
    var idDatabin : String
    let baseURL = \"https://datadrop.wolframcloud.com/api/v1.0/\"
    
    init(idDatabin: String) {
        self.idDatabin = idDatabin
    }
"

swiftDataDropRecentFunction = "
func getRecent(completionHandler: (JSON -> Void)) {
        var getURL = baseURL +  \"Recent?bin=\" + idDatabin + \"&_exportform=JSON\"
        
        Alamofire.request(.GET, getURL).responseJSON { (_,_,json,_) in
            dispatch_async(dispatch_get_main_queue(), {
                let jsonObj = JSON(json!)
                completionHandler(jsonObj)
            })
        }
    }
"

swiftDataDropAddFunction = "
func addData(newData: [String: String],completionHandler: (JSON -> Void)) {
        var sNewData = \"\"
        var first = true
        
        for (key,value) in newData {
            if first {
                first = false
            } else {
                sNewData += \"%26\"
            }
            sNewData += key + \"=\" + value
        }
        
        var addUrl = baseURL + \"Add?bin=\" + idDatabin + \"&Data=\" + sNewData + \"&_exportform=JSON\"
        
        Alamofire.request(.GET, addUrl).responseJSON { (_,_,json,_) in
            dispatch_async(dispatch_get_main_queue(), {
                let jsonObj = JSON(json!)
                completionHandler(jsonObj)
            })
        }
    }
"

swiftAuxiliarFunctions = ""

swiftFooter = "}
"