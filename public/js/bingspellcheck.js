var Worker = require("tiny-worker");
let host = 'api.cognitive.microsoft.com';
let path = '/bing/v7.0/spellcheck';
let key = '24c850edb1fe46ec82bb0befa3e285aa';
let mkt = "en-US";
let mode = "spell";
let query_string = "?mkt=" + mkt + "&mode=" + mode;
let https = require('https');
let text = "";

let response_handler = function(response) {
	let body = '';
	response.on('data', function(d) {
		body += d;
	});
	response.on('end', function() {
		let body_ = JSON.parse(body);
		var newQuestion = text;
		body_.flaggedTokens.forEach((element) => {
			newQuestion = newQuestion.replace(element.token, element.suggestions[0].suggestion)
		});
		postMessage(newQuestion)
	});
	response.on('error', function(e) {
		console.log('Error: ' + e.message);
	});
};

onmessage = function findBingSpellCheck(input) {
	text = input.data[0];
	let request_params = {
		method: 'POST',
		hostname: host,
		path: path + query_string,
		headers: {
			'Content-Type': 'application/x-www-form-urlencoded',
			'Content-Length': text.length + 5,
			'Ocp-Apim-Subscription-Key': key,
		}
    };
		let req = https.request(request_params, response_handler);
		req.write("text=" + text);
		req.end();
	
}