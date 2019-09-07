const os = require('os')
var osvers = os.platform()

var Worker = require("tiny-worker");

onmessage = function findTextAnswer(input) {
	var request = require("request");
	console.log(input)
	var options = {
		method: 'POST',
		url: 'http://159.89.39.148:8080',
		formData: {
			text: input.data[0],
			question: input.data[1],
			format: input.data[3],
			number: input.data[2]
		}
	};

	request(options, function(error, response, body) {
		if (error) throw new Error(error);
		postMessage(JSON.parse(body).text);
	});
}