var java = require('java');
const os = require('os')
var osvers = os.platform()
const etudeFilepath = __dirname.replace("/public/js","").replace("\\public\\js","")
if(osvers == "darwin"){
	java.classpath.push(etudeFilepath + "/MacKernel.jar");
	java.classpath.push(etudeFilepath + "/Contents/Resources/Wolfram\ Player.app/Contents/SystemFiles/Links/JLink/JLink.jar");
} else {
	java.classpath.push(etudeFilepath + "/WindowsKernel.jar");
	java.classpath.push(etudeFilepath + "/12.0/SystemFiles/Links/JLink/JLink.jar");
}
var kernel = java.newInstanceSync('p1.Kernel', etudeFilepath);

var Worker = require("tiny-worker");

onmessage = function findTextAnswer(input) {
	console.log(input)
	var results = kernel.findTextAnswerSync(input.data[0], input.data[1], input.data[2], input.data[3]);
	postMessage(results);
}