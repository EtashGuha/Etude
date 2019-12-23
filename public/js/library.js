const {
	dialog
} = require('electron').remote;
const { shell } = require('electron')
const Store = require('electron-store');
const windowFrame = require('electron-titlebar')
var store = new Store();
var path = require("path")
const {
	ipcRenderer
} = require('electron');
const date = require('date-and-time');
const now = new Date();
const remote = require('electron').remote;
var win = remote.BrowserWindow.getFocusedWindow();
var currSet;
console.log(store.store)
if(store.has("libraryStore")){
	currSet = new Set(store.get("libraryStore"))
} else {
	currSet = new Set();
}
console.log(require('root-require')('package.json').version);
var i = store.size;
var __PDF_DOC,
	__CURRENT_PAGE,
	__TOTAL_PAGES,
	__PAGE_RENDERING_IN_PROGRESS = 0,
	index = 0;
var data = ipcRenderer.sendSync('get-file-data')
console.log(remote.getGlobal('sharedObject').newWindow)
var numDaysLeft = date.subtract(now, new Date(store.get("startDate"))).toDays()
console.log(numDaysLeft)
if (data ===  null || remote.getGlobal('sharedObject').newWindow) {
    console.log("There is no file")
} else {
    // Do something with the file.
    	ipcRenderer.send('show_pdf_message', data);
		window.location.href = 'summarizing.html';
}
var counter = 0;
currSet.forEach(function(value) {
	
		show_nextItem(value, counter.toString());
		showPDF_fresh(value, counter);
		counter = counter + 1;
	
});


document.getElementById('stripeIDBlock').innerHTML = store.get("stripeID");
document.getElementById('myButton').addEventListener('click', () => {
	dialog.showOpenDialog({
		properties: ['openFile'], // set to use openFileDialog
		filters: [{
			name: "PDFs",
			extensions: ['pdf']
		}] // limit the picker to just pdfs
	}, (filepaths) => {
		var filePath = filepaths[0];
		if(!currSet.has(filePath)){
			currSet.add(filePath)
			console.log(Array.from(currSet))
			store.set("libraryStore", Array.from(currSet))
			console.log(store.store)
			show_nextItem(filePath, i.toString());
			showPDF(filePath);
		}
	})

})

document.getElementById('closeButton').addEventListener('click', () => {
    win.close();

})

document.getElementById('etudeButton').addEventListener('click', () => {
	shell.openExternal('https://www.etudereader.com')

})




function showPDF(pdf_url) {
	PDFJS.getDocument({
		url: pdf_url
	}).then(function(pdf_doc) {
		__PDF_DOC = pdf_doc;
		__TOTAL_PAGES = __PDF_DOC.numPages;
		showPage(1);
	}).catch(function(error) {
		alert(error.message);
	});;
}

function showPDF_fresh(pdf_url, num) {
	PDFJS.getDocument({
		url: pdf_url
	}).then(function(pdf_doc) {
		__PDF_DOC = pdf_doc;
		__TOTAL_PAGES = __PDF_DOC.numPages;
		showPage_fresh(1, num);
	}).catch(function(error) {
		alert(error.message);
	});;
}

function showPage(page_no) {
	__PAGE_RENDERING_IN_PROGRESS = 1;
	__CURRENT_PAGE = page_no;

	// Fetch the page
	__PDF_DOC.getPage(page_no).then(function(page) {
		var __CANVAS = $('.pdf-canvas').get($(".pdf-canvas").length - 1),
			__CANVAS_CTX = __CANVAS.getContext('2d');
		var scale_required = __CANVAS.width / page.getViewport(1).width;

		// Get viewport of the page at required scale
		var viewport = page.getViewport(scale_required);

		// Set canvas height
		__CANVAS.height = viewport.height;

		var renderContext = {
			canvasContext: __CANVAS_CTX,
			viewport: viewport
		};

		// Render the page contents in the canvas
		page.render(renderContext).then(function() {
			__PAGE_RENDERING_IN_PROGRESS = 0;
			$(".pdf-canvas").show();

		});
		// index++;
	});
}

function showPage_fresh(page_no, num) {
	__PAGE_RENDERING_IN_PROGRESS = 1;
	__CURRENT_PAGE = page_no;


	// Fetch the page
	__PDF_DOC.getPage(page_no).then(function(page) {
		var __CANVAS = $('.pdf-canvas').get(num),
			__CANVAS_CTX = __CANVAS.getContext('2d');
		var scale_required = __CANVAS.width / page.getViewport(1).width;

		// Get viewport of the page at required scale
		var viewport = page.getViewport(scale_required);

		// Set canvas height
		__CANVAS.height = viewport.height;

		var renderContext = {
			canvasContext: __CANVAS_CTX,
			viewport: viewport
		};

		// Render the page contents in the canvas
		page.render(renderContext).then(function() {
			__PAGE_RENDERING_IN_PROGRESS = 0;

			$(".pdf-canvas").show();
		});
	});
}

function show_nextItem(pdf_path, removeWhich) {
	// var name = pdf_path.split('/');
	// we get the name of the pdf
	var filenamewithextension = path.parse(pdf_path).base;
	console.log(pdf_path)
	pdf_pathNext = replaceAll(pdf_path, " ", "?*?")
	console.log(pdf_pathNext)
	var filename = filenamewithextension.split('.')[0];
	var next_text = "<div class='col-md-2 book_section'><div><center><canvas class='pdf-canvas' data ='" + pdf_pathNext + "' id = 'viewer'></canvas><img class = 'minusImage' data =" + pdf_pathNext + " id = 'myButton' src='./public/images/cross.png'/><p style = 'width: 200px; word-break: break-all;'>" + filename + "</p></center></div></div>";
	var next_div = document.createElement("div");
	next_div.innerHTML = next_text;
	console.log(next_div.innerHTML)
	document.getElementById('container').append(next_div);
}

function replaceAll(str, find, replace) {
    return str.replace(new RegExp(escapeRegExp(find), 'g'), replace);
}

function escapeRegExp(str) {
    return str.replace(/([.*+?^=!:${}()|\[\]\/\\])/g, "\\$1");
}

//when the user select the pdf
$(document).on("click", ".pdf-canvas", function() {
	console.log($(this).attr("data"));
	console.log("above you clicked sth");
	var realPDFPath = replaceAll($(this).attr("data"), "?*?", " ")
	ipcRenderer.send('show_pdf_message', realPDFPath);
	window.location.href = 'summarizing.html';
});
// when the user click the minus button
$(document).on("click", ".minusImage", function() {
	($(this).parent()).parent().parent().remove();
	var pdf_path = ($(this).attr("data"));
	pdf_path = replaceAll(pdf_path, "?*?", " ")
	//delete it in the store
	currSet.delete(pdf_path)
	console.log(currSet)
	store.set("libraryStore", Array.from(currSet))
});