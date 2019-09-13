const {
	dialog
} = require('electron').remote;
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
const electron = require("electron")
const userDataPath = (electron.app || electron.remote.app).getPath('userData');
console.log("USER DATA PATH: " + userDataPath)
var tools = require('./createFile/coordinates.js')
var bookmarkArray = [];
var Tokenizer = require('sentence-tokenizer');
var tokenizer = new Tokenizer('Chuck');
const viewerEle = document.getElementById('viewer');
viewerEle.innerHTML = ''; // destroy the old instance of PDF.js (if it exists)
const iframe = document.createElement('iframe');
iframe.src = path.resolve(__dirname, `./pdfjsOriginal/web/viewer.html?file=${require('electron').remote.getGlobal('sharedObject').someProperty}`);
console.log(iframe.src)
const etudeFilepath = __dirname.replace("/public/js", "").replace("\\public\\js", "")
const secVersionFilepath = userDataPath + "/folderForHighlightedPDF/secVersion.pdf"

viewerEle.appendChild(iframe);

filepath = require('electron').remote.getGlobal('sharedObject').someProperty;
deepai.setApiKey('a5c8170e-046a-4c56-acb1-27c37049b193');
//get text from pdf to send to flask backends
var PDF_URL = filepath;
var capeClicked = false;
var btnClicked = false;
var bookmarkOpened = false;
var htmlForEntireDoc = ""
// pdfAllToHTML(PDF_URL);
var pdfToHtmlWorker = new Worker(etudeFilepath + "/public/js/pdfToHtml.js");
var kernelWorker = new Worker(etudeFilepath + "/public/js/kernel.js")
var updateHighlightsWorker = new Worker(etudeFilepath + "/public/js/updateHighlights.js")
pdfToHtmlWorker.onmessage = function(ev) {
	console.log(ev);
	pdfToHtmlWorker.terminate();
};

console.log("hello")
console.log(userDataPath)
console.log(etudeFilepath)
console.log(PDF_URL);
pdfToHtmlWorker.postMessage([PDF_URL, userDataPath, etudeFilepath]);
console.log("has")
var numPages = 0;
PDFJS.getDocument({
	url: PDF_URL
}).then(function(pdf_doc) {
	__PDF_DOC = pdf_doc;
	numPages = __PDF_DOC.numPages;
});


$("#bookmark_icon").click(function() {
	//get the page number
	var whichpagetobookmark = document.getElementsByTagName('iframe')[0].contentWindow.document.getElementById('pageNumber').value;
	for (var j = 0; j < bookmarkArray.length; j++) {
		if (bookmarkArray[j] == whichpagetobookmark)
			return;
	}
	show_nextItem(whichpagetobookmark, null);
	//store the page number in the database
	bookmarkArray.push(whichpagetobookmark);
	showPDF(filepath, parseInt(whichpagetobookmark));
})

function showPDF(pdf_url, bookmark_page) {
	//PDFJS.GlobalWorkerOptions.workerSrc ='../../node_modules/pdfjs-dist/build/pdf.worker.js';
	PDFJS.getDocument({
		url: pdf_url
	}).then(function(pdf_doc) {
		__PDF_DOC = pdf_doc;
		__TOTAL_PAGES = __PDF_DOC.numPages;
		// Show the first page
		showPage(bookmark_page);
		// store.set(i.toString(),pdf_url);
		// i++;
	}).catch(function(error) {
		alert(error.message);
	});
}

function showPage(page_no) {
	__PAGE_RENDERING_IN_PROGRESS = 1;
	__CURRENT_PAGE = page_no;

	// Fetch the page
	__PDF_DOC.getPage(page_no).then(function(page) {
		var __CANVAS = $('.bookmark-canvas').get($(".bookmark-canvas").length - 1),
			__CANVAS_CTX = __CANVAS.getContext('2d');
		// const viewerEle = document.getElementsByClassName('pdf-canvas')[$(".pdf-canvas").length-1];
		// As the canvas is of a fixed width we need to set the scale of the viewport accordingly
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

			$(".bookmark-canvas").show();
			$(".deleteImage_").show();

		});
	});
}

function show_nextItem(whichpage, removeWhich) {
	var next_text = "<div class='bookmark_section col-md-2 col-lg-2' style = 'margin-top:10px;'><div><canvas class='bookmark-canvas' data = '" + whichpage + "'></canvas><img class = 'deleteImage_' src='public/images/bookmarkminus.png' data = '" + removeWhich + "' id = 'myButton'/></div></div>";
	var next_div = document.createElement("div");
	next_div.innerHTML = next_text;
	document.getElementById('bookmark_container').append(next_div);
}
// delete the bookmark in the bookmark page
$(document).on("click", ".deleteImage_", function() {
	($(this).parent()).parent().remove();

});
// when the user click the bookmark
$(document).on("click", ".bookmark-canvas", function() {
	console.log(document.getElementsByTagName('iframe')[0].contentWindow.document.getElementById('thumbnailView').children)
	console.log(document.getElementsByTagName('iframe')[0].contentWindow.document.getElementById('thumbnailView').children[$(this).attr("data") - 1])
	document.getElementsByTagName('iframe')[0].contentWindow.document.getElementById('thumbnailView').children[$(this).attr("data") - 1].click()
});

var searchbox = document.getElementById("questionVal");
// Execute a function when the user releases a key on the keyboard
searchbox.addEventListener("keyup", function(event) {
  // Number 13 is the "Enter" key on the keyboard
  if (event.keyCode === 13) {
    // Cancel the default action, if needed
    // Trigger the button element with a click
    document.getElementById("cape_btn").click();
  }
});

$("#cape_btn").click(function() {
	kernelWorker = new Worker(etudeFilepath + "/public/js/kernel.js")
	updateHighlightsWorker = new Worker(etudeFilepath + "/public/js/updateHighlights.js")
	console.log("Cape button clicked")
	if (capeClicked) {
		document.getElementById("myDropdown").classList.toggle("show");
	}
	setTimeout(function() {
		if (htmlForEntireDoc == "") {
			htmlForEntireDoc = htmlWholeFileToPartialPlainText(1, numPages);
		}
		htmlForEntireDoc.then((x) => {
			var promiseToAppend = new Promise(function(resolve, reject) {
				console.log("beginning promise")
				kernelWorker.onmessage = function(ev) {
					$("#capeResult").empty().append(ev.data[0] + " <hr style=\"margin-top: 15px; margin-bottom: 15px\"> " + ev.data[1]);
					updateHighlights(ev.data)
					kernelWorker.terminate()
					resolve("GOOD")
				}
				console.log("redefined kernelWorker on message")
				kernelWorker.postMessage([x, $("#questionVal").val(), "2", "Sentence"])
				console.log("kernel worker put up")
				//kernel.findTextAnswerSync();

			});
			//$("#capeResult").empty().append(kernel.findTextAnswerSync(x, $("#questionVal").val(), 2, "Sentence"));
			//document.getElementById("myDropdown").classList.toggle("show");
			promiseToAppend.then((data) => {
				console.log(data)
				document.getElementById("myDropdown").classList.toggle("show");
				document.getElementById('searchloader').style.display = 'none';
				document.getElementById('searchbuttonthree').style.color = 'black';
				document.getElementById('cape_btn').style.backgroundColor = '';
				console.log("Showing")
			});
		});
		capeClicked = true;
	}, 1);
});

$("#help").click(function() {
	// close the bookmark page
	$("#bookmark_item").attr("data", "true");
	$("#bookmark_item").click();

	$('.help_popup').show();
	$('.help_popup').click(function() {
		$('.help_popup').hide();
	});
})

//summarization function
$('#summarizingButton').click(function() {
	$('.su_popup').hide();
	summaryButtonPressed($('#pageRange').val(), $('#topageRange').val());
	// here you can add the loading button
	$('.summarizer_loading').show();
	// $('.hover_bkgr_fricc').click(function(){
	//       $('.hover_bkgr_fricc').hide();
	//   });
})

$('#escapeSUPopupButton').click(function() {
	$('.su_popup').hide();
})

var textDsum = "";
var iPagesum = 0;
var iEndPagesum = 0;

function processSummarizationResult(t) {
	console.log(t)
	noLineBreakText = t["output"].replace(/(\r\n|\n|\r)/gm, " ");
	tokenizer.setEntry(noLineBreakText);
	updateHighlights(tokenizer.getSentences())
	$('.summarizer_loading').hide();
};

function summaryButtonPressed(firstpage, lastpage) {
	var htmlStuff = htmlWholeFileToPartialPlainText(firstpage, lastpage);
	htmlStuff.then((x) => {
		deepai.callStandardApi("summarization", {
			text: x
		}).then((resp) => processSummarizationResult(resp))
	});
}

function htmlWholeFileToPartialPlainText(firstpage, lastpage) {
	return new Promise(function(resolve, reject) {
		var filenamewithextension = path.parse(PDF_URL).base;
		filenamewithextension = filenamewithextension.split('.')[0];
		var outputfile = userDataPath + '/tmp/' + filenamewithextension + '.html';
		const htmlToJson = require('html-to-json');
		let bigarray = [];
		let bigarrayback = [];
		//the correct html file directory within our project
		console.log(outputfile)
		fs.readFile(outputfile, "utf8", function(err, data) {
			let datadata = data.split("<div class=\"page\"");
			let newstring = "";
			for (let i = firstpage; i <= lastpage; i++) {
				if (i <= datadata.length) {
					newstring += datadata[i];
				}
			}
			//console.log(newstring);
			let newdata = newstring.split("<div class=\"p\"");
			newdata.shift();
			newdata.forEach(function(item) {
				item = item.substring(10 + item.search("font-size"));
				let valued = item.substring(item.search(">") + 1, item.search("<"));
				let index = (item.substring(0, item.search("pt")));

				function checkSize(age) {
					return age == index;
				}
				if (bigarrayback.findIndex(checkSize) == -1) {
					bigarrayback.push(index);
					valued += " ";
					bigarray.push(valued);
				} else {
					valued += " ";
					bigarray[bigarrayback.findIndex(checkSize)] += valued;
				}
			});
			let maxindex = -1;
			let max = 0;
			bigarray.forEach(function(thing, index) {
				if (thing.length > max) {
					maxindex = index;
					max = thing.length;
				}
			});

			resolve(bigarray[maxindex]);
		});

	});
}

$('#getRangeButton').click(function() {
	//close the bookmark page
	$("#bookmark_item").attr("data", "true");
	$("#bookmark_item").click();

	//$('#getRangeButton').hide();
	$('.su_popup').show();
})

function jumpPage(pageNumber) {
	if (document.getElementsByTagName('iframe')[0].contentWindow.document.getElementById('thumbnailView') != null &&
		typeOf(document.getElementsByTagName('iframe')[0].contentWindow.document.getElementById('thumbnailView').childNodes[pageNumber - 1]) != "text" &&
		document.getElementsByTagName('iframe')[0].contentWindow.document.getElementById('thumbnailView').childNodes[pageNumber - 1] != undefined) {
		document.getElementsByTagName('iframe')[0].contentWindow.document.getElementById('thumbnailView').childNodes[pageNumber - 1].click()
	} else {
		window.setTimeout(jumpPage, 100, pageNumber)
	}
}
function updateHighlights(arr){
	console.log(arr)
	var searchQueries = ""
	arr.forEach((item, index) => {
		item = replaceAll(item,"\u00A0", "%3D");
		item = replaceAll(item, " ", "%3D")
		searchQueries += "%20" + item
	})

	searchQueries = searchQueries.substring(3)
	iframe.src = path.resolve(__dirname, `./pdfjsOriginal/web/viewer.html?file=${require('electron').remote.getGlobal('sharedObject').someProperty}#search=${searchQueries}`);
	viewerEle.appendChild(iframe);
}

function replaceAll(str, find, replace) {
    return str.replace(new RegExp(escapeRegExp(find), 'g'), replace);
}

function escapeRegExp(str) {
    return str.replace(/([.*+?^=!:${}()|\[\]\/\\])/g, "\\$1");
}

function changePage() {
	viewerEle.innerHTML = "";
	console.log(`./pdfjsOriginal/web/viewer.html?file=${secVersionFilepath}`)
	iframe.src = path.resolve(__dirname, `./pdfjsOriginal/web/viewer.html?file=${secVersionFilepath}`);
	console.log(iframe)
	viewerEle.appendChild(iframe);
}