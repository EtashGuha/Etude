var fs = require("fs");
var request = require("request");
var type = require('type-of')
var options = {
	method: 'POST',
	encoding: null,
	url: 'http://127.0.0.1:5000/file-upload',
	formData: {
		file: {
			value: fs.createReadStream('/Users/etashguha/Downloads/ocrscan.pdf'),
			options: {
				filename: 'ocrscan.pdf',
				contentType: null
			}
		}
	}
};

request(options, function(error, response, body) {
	if (error) throw new Error(error);
	console.log(response)
	fs.writeFileSync("/Users/etashguha/Desktop/pleaseword.pdf", body)
});