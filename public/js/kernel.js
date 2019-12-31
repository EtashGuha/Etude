const os = require('os')
const ex = require('C:/Users/alimi/Downloads/EtudeXML/Software/beta8/Etude/public/js/ex')
var osvers = os.platform()

var Worker = require("tiny-worker");

onmessage = function findTextAnswer(input) {
	console.log("at least here dawg")
	const result = ex.getAnswer(input.data[1], input.data[0])
	console.log(result)
	result.then((data)=> {
		console.log("data")
		console.log(data)
		postMessage(data);
	})
}