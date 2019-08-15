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
    console.log("Herar")
    console.log(typeof(input.data[0]))
    var exec = require('child_process').exec, child;
    const etudeFilepath = __dirname.replace("/public/js","").replace("\\public\\js","")
    var filenamewithextension = path.parse(input.data[0]).base;
    filenamewithextension = filenamewithextension.split('.')[0];
    console.log(filenamewithextension)
    //update directory to JAR file
    var pathOfFile = input.data[1] + '/tmp/' + filenamewithextension + '.html'
    try {
      if (fs.existsSync(pathOfFile)) {
        console.log("done")
        postMessage("html exists already")
        return;
      }
    } catch(err) {
      console.error(err)
    }
    var executionstring;
    if(osvers == "win32"){
      executionstring = '\"%ProgramFiles%/Java/jdk-11.0.1/bin/java\" -jar \"' + etudeFilepath + '/PDFToHTML.jar\" \"' + input[0] + '\" \"' + input.data[1] + '/tmp/' + filenamewithextension + '.html\"';
      console.log(executionstring)
    } else {
      console.log("mac")
      executionstring = 'java -jar ' + etudeFilepath + '/PDFToHTML.jar \"' + input.data[0] + '\" \"' + input.data[1] +  '/tmp/' + filenamewithextension + '.html\"';  
    }
    console.log(executionstring)
    child = exec(executionstring,
      function (error, stdout, stderr) {
        console.log("nneare")
        console.log(error)
        postMessage("done")
        if (error != null) {
            console.log("done")
           console.log('exec error: ' + error);
           postMessage("done")
        }
      });
  };



