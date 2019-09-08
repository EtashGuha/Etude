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
var Worker = require("tiny-worker");





onmessage = function pdfToHtml(input) {
	const etudeFilepath = input.data[2];	
	var unpackedDirectory = etudeFilepath.replace("app.asar", "app.asar.unpacked")
	var jre = require(unpackedDirectory + "/node_modules/node-jre");
	var filenamewithextension = path.parse(input.data[0]).base;
	filenamewithextension = filenamewithextension.split('.')[0];

	//update directory to JAR file
	var pathOfFile = input.data[1] + '/tmp/' + filenamewithextension + '.html'
	try {
		if (fs.existsSync(pathOfFile)) {
			console.log("done")
			postMessage("html exists already")
			return;
		}
	} catch (err) {
		console.error(err)
	}

	console.log(input.data[1] + '/tmp/ '+ filenamewithextension + '.html')
	console.log("trying")
	var output = jre.spawnSync(  // call synchronously
	    [unpackedDirectory+'/PDFToHTML.jar'],                // add the relative directory 'java' to the class-path
	    'org.fit.pdfdom.PDFToHTML',                 // call main routine in class 'Hello'
	    [input.data[0],input.data[1] + '/tmp/ '+ filenamewithextension + '.html'],               // pass 'World' as only parameter
	    { encoding: 'utf8' }     // encode output as string
  	).stdout.trim();           // take output from stdout as trimmed String
	postMessage("done")
};