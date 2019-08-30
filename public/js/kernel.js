const os = require('os')
var osvers = os.platform()

var Worker = require("tiny-worker");

onmessage = function findTextAnswer(input) {
	etudeFilepath = input.data[4];
	var unpackedDirectory = etudeFilepath.replace("app.asar", "app.asar.unpacked")
	var java = require(unpackedDirectory + '/node_modules/java');
	console.log(unpackedDirectory + '/node_modules/java')
	if (osvers == "darwin") {
		console.log(unpackedDirectory + '/MacKernel.jar')
		java.classpath.push(unpackedDirectory + '/MacKernel.jar');
		java.classpath.push(unpackedDirectory + "/WolframContents/Resources/Wolfram\ Player.app/Contents/SystemFiles/Links/JLink/JLink.jar");
	} else {
		java.classpath.push(etudeFilepath + "/WindowsKernel.jar");
		java.classpath.push(etudeFilepath + "/12.0/SystemFiles/Links/JLink/JLink.jar");
	}
	var kernel = java.newInstanceSync('p1.Kernel', unpackedDirectory);
	console.log(input)
	var results = kernel.findTextAnswerSync(input.data[0], input.data[1], input.data[2], input.data[3]);
	postMessage(results);
}