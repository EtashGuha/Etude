const fs = require('fs');
const path = require('path');
const express = require('express');
const bodyParser = require('body-parser');
const opn = require('opn');
const ip = require('ip');

const app = express();

const options = process.argv.reduce((acc, arg) => {
	const pair = arg.split('=');
	if (pair.length === 2) {
		acc[pair[0]] = pair[1];
	}
	return acc;
}, {});

const handleAnnotation = (req, res, handler) => {
	const dir = path.resolve(__dirname, 'annotations');
	if (!fs.existsSync(dir)){
    fs.mkdirSync(dir);
	}

	handler(dir);
	res.end();
};

app.use(bodyParser.json()); // for parsing application/json
app.use(bodyParser.urlencoded({ extended: true })); // for parsing application/x-www-form-urlencoded

app.get('/annotations', (req, res) => {
	handleAnnotation(req, res, (dir) => {
		const xfdfFile = (req.query.did) ? path.resolve(dir, req.query.did + '.xfdf') : path.resolve(dir, 'default.xfdf');
		if (fs.existsSync(xfdfFile)) {
			res.header('Content-Type', 'text/xml');
			res.send(fs.readFileSync(xfdfFile));
		} else {
			res.status(204);
		}
	});
});

app.post('/annotations', (req, res) => {
	handleAnnotation(req, res, (dir) => {
		const xfdfFile = (req.body.did) ? path.resolve(dir, req.body.did + '.xfdf') : path.resolve(dir, 'default.xfdf');
		try {
			res.send(fs.writeFileSync(xfdfFile, req.body.data));
		} catch(e) {
			res.status(500);
		}
	});
	res.end();
});

app.get('/ip', (req, res) => {
	res.send(ip.address());
});

app.get('/', (req, res) => {
	res.redirect('/samples');
});

app.get('/doc', (req, res) => {
	res.redirect('/doc/PDFTron.WebViewer.html');
});

app.get('/doc/index.html', (req, res) => {
	res.redirect('/doc/PDFTron.WebViewer.html');
});

app.get(/\/samples(.*)(\/|\.html)$/, (req, res, next) => {
	if (req.query.key || !options.key) {
		next();
	} else {
		if (req.originalUrl.slice(-10) === 'index.html') {
			res.redirect(`${req.originalUrl}?key=${options.key}`)
		} else {
			res.redirect(`./index.html?key=${options.key}`)
		}
	}
});

app.use(express.static(path.resolve(__dirname), {
	setHeaders: (res) => {
    res.set('Service-Worker-Allowed', '/')
  }
}));

app.get('/config.js', (req, res) => {
	res.send('./config.js');;
})

app.listen(3000, '0.0.0.0', (err) => {
	if (err) {
		console.error(err);
	} else {
		console.info(`Listening at localhost:3000 (http://${ip.address()}:3000)`);
		if (process.argv[2] === 'samples') {
			opn(`http://localhost:3000/samples`);
		} else if (process.argv[2] === 'doc') {
			opn('http://localhost:3000/doc/PDFTron.WebViewer.html');
		}
	}
});