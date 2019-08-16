var Worker = require("tiny-worker");
const thread = require("threads/worker")
const path = require('path');
const log = require('electron-log');
const fs = require('fs');
var jre = require('node-jre')
log.info('Hello, log for the first time');

var typeOf = require('typeof');
const os = require('os')
var osvers = os.platform()
console.log(osvers)
var textData = null;
var bookmarkArray = [];
const etudeFilepath = __dirname.replace("/public/js","").replace("\\public\\js","")
var tools = require(etudeFilepath + '/public/js/createFile/coordinates.js')
var viewerEle;
var iframe;
var secVersionFilepath;
var objectsJsonFilePath;
onmessage = function updateHighlights(input) {
	secVersionFilepath = input.data[3] + "/folderForHighlightedPDF/secVersion.pdf"
	objectsJsonFilePath = input.data[3] + "/tmp/object.json";

	console.log("running")
	console.log(input)
	console.log('asdfasdf')
	if(fs.existsSync(secVersionFilepath)){
		fs.unlinkSync(secVersionFilepath);
  	}
  	console.log("nana")
	tools.extractor(input.data[0],input.data[2], secVersionFilepath, input.data[3]);
	checkFlag(input.data[1]);
}

function checkForJump(){
	console.log("checking for jump")
	if(!fs.existsSync(objectsJsonFilePath)){
		setTimeout(() => {checkForJump()}, 100);
	} else {
		try {
			var jsonContents = getJsonContents()
			var answer = jsonContents[0]['page'] + 1;
			postMessage(answer);
		} catch (err){
			setTimeout(() => {checkForJump()}, 100);
		}
	}
}

function checkFlag(isSearch) {
	console.log("what is up")
	console.log(secVersionFilepath)
	if(!fs.existsSync(secVersionFilepath)){
	  console.log("checking")
	  setTimeout(() => {checkFlag(isSearch)}, 100); /* this checks the flag every 100 milliseconds*/
	} else {
	  if(isSearch){
	  	console.log("CHECKINGFORJUMP")
	  	checkForJump()
	  } else {
	  	postMessage("all good")
	  }
	}
}

function getJsonContents(){
	try {
		var contents = fs.readFileSync(objectsJsonFilePath);
		console.log("naaa")
		var jsonContents = JSON.parse(contents)
		return jsonContents;
	} catch (err){
		window.setTimeout(getJsonContents, 100)
	}
}