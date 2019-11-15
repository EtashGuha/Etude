const {
	dialog
} = require('electron').remote;
const { shell } = require('electron')
var HashMap = require('hashmap');
var map = new HashMap();
const remote = require('electron').remote;
const Store = require('electron-store');
const store = new Store();
var win = remote.BrowserWindow.getFocusedWindow();
const path = require('path');
const log = require('electron-log');
const fs = require('fs');
log.info('Hello, log for the first time');
var typeOf = require('typeof');
const os = require('os')
const windowFrame = require('electron-titlebar')
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
console.log("Entering summarinzg js")
console.log(require('electron').remote.getGlobal('sharedObject').someProperty)
iframe.src = path.resolve(__dirname, `./pdfjsOriginal/web/viewer.html?file=${require('electron').remote.getGlobal('sharedObject').someProperty}`);
console.log(iframe.src)
const etudeFilepath = __dirname.replace("/public/js", "").replace("\\public\\js", "")
const secVersionFilepath = userDataPath + "/folderForHighlightedPDF/secVersion.pdf"
viewerEle.appendChild(iframe);
var currArr;
filepath = require('electron').remote.getGlobal('sharedObject').someProperty;
console.log(deepai)
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
document.getElementById("getRangeButton").disabled = true;
document.getElementById("getRangeButton").style.opacity = 0.5;
document.getElementById("cape_btn").disabled = true;
document.getElementById("searchParent").style.opacity = 0.5;
document.getElementById("questionVal").disabled = true;
document.getElementById('searchloader').style.display = 'block';
document.getElementById('searchbuttonthree').style.color = 'white';
document.getElementById('cape_btn').style.backgroundColor = 'white';

var textForEachPage;

pdfToHtmlWorker.onmessage = function(ev) {
	enableEtude();
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
console.log(numPages)

const {
	ipcRenderer
} = require('electron');



function enableEtude() {
	document.getElementById("getRangeButton").disabled = false;
	document.getElementById("getRangeButton").style.opacity = 1.0;
	document.getElementById("cape_btn").disabled = false;
	document.getElementById("searchParent").style.opacity = 1.0;
	document.getElementById("questionVal").disabled = false;
	document.getElementById('searchloader').style.display = 'none';
	document.getElementById('searchbuttonthree').style.color = 'black';
	document.getElementById('cape_btn').style.backgroundColor = '';

}

$("#searchToggle").click(function() {
	console.log(document.getElementById('searchParent').style.visibility)
	if(document.getElementById('searchParent').style.visibility === 'hidden') {
		document.getElementById('searchParent').style.visibility = 'visible';
		iframe.contentWindow.closeFindBar()
	} else {
		document.getElementById('searchParent').style.visibility = 'hidden';
		iframe.contentWindow.openFindBar()
	}
	
});

enableEtude()

$("#bookmark_icon").click(function() {
	//get the page number
	var whichpagetobookmark = document.getElementsByTagName('iframe')[0].contentWindow.document.getElementById('pageNumber').value;
	for (var j = 0; j < bookmarkArray.len; j++) {
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

// var closeButton = document.getElementById("closeButton");
// closeButton.addEventListener('click', () => {
//     win.close();
// })

// var maxbuttton = document.getElementById("maxbutton");
// maxbuttton.addEventListener('click', () => {
//     win.maximize();
// })

// var minbutton = document.getElementById("minbutton");
// minbutton.addEventListener('click', () => {
//     win.minimize();
// })



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

searchbox.addEventListener("click", function() {
	document.getElementById("myDropdown").classList.add("show");
});

var closeSearch = document.getElementById("closesearchbutton");
closeSearch.addEventListener("click", function() {
	document.getElementById("myDropdown").classList.remove("show");
})
// var body = document.getElementsByTagName("BODY")[0];
// var except = document.getElementById("myDropdown");
// body.addEventListener("click", function () {
// 	document.getElementById("myDropdown").classList.toggle("show");
// }, false);
// except.addEventListener("click", function (ev) {
//     ev.stopPropagation(); //this is important! If removed, you'll get both alerts
// }, false);

$("#cape_btn").click(function() {
	kernelWorker = new Worker(etudeFilepath + "/public/js/kernel.js")
	updateHighlightsWorker = new Worker(etudeFilepath + "/public/js/updateHighlights.js")
	console.log("Cape button clicked")
	document.getElementById("myDropdown").classList.toggle("show");
	setTimeout(function() {
		console.log(getNumPages())
		var getpdftext = getPDFText(1, getNumPages())
		getpdftext.then((x) => {
			var promiseToAppend = new Promise(function(resolve, reject) {
				console.log("beginning promise")
				kernelWorker.onmessage = function(ev) {
					$("#capeResult").empty().append(ev.data[0]);
					updateHighlights(ev.data)
					console.log("refreshed");
					if(document.getElementById("myDropdown").classList.contains("show")){
						console.log("Not showing dropdown");
						document.getElementById("myDropdown").classList.toggle("show");
					}
					kernelWorker.terminate()
					resolve("GOOD")
				}
				console.log("redefined kernelWorker on message")
				kernelWorker.postMessage([x, $("#questionVal").val(), "8", "Sentence"])
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

$("#etudeButton").click(function() {
//document.getElementById('etudeButton').addEventListener('click', () => {
	console.log("Attempting to go to website");
	goToWebsite();

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


function goToWebsite() {
	shell.openExternal('https://www.etudereader.com')
}
function processSummarizationResult(t) {
	console.log("here we are")
	console.log(t)
	noLineBreakText = t["output"].replace(/(\r\n|\n|\r)/gm, " ");
	tokenizer.setEntry(noLineBreakText);
	updateHighlights(tokenizer.getSentences())
	$('.summarizer_loading').hide();
};

function summaryButtonPressed(firstpage, lastpage) {
	var getpdftext = getPDFText(firstpage, lastpage)
	getpdftext.then((x) => {
		console.log(x);
		deepai.callStandardApi("summarization", {
			text: x
		}).then((resp) => processSummarizationResult(resp))
	});
}


$('#getRangeButton').click(function() {
	//close the bookmark page
	$("#bookmark_item").attr("data", "true");
	$("#bookmark_item").click();

	//$('#getRangeButton').hide();
	$('.su_popup').show();
})

function updateHighlights(arr){
	console.log(arr)
	currArr = arr;
	var searchQueries = ""
	arr.forEach((item, index) => {
		item = item.replace(/[^a-zA-Z ]/g, "")
		item = replaceAll(item,"\u00A0", "%3D");
		item = replaceAll(item, " ", "%3D")
		searchQueries += "%20" + item
	})

	searchQueries = searchQueries.substring(3)
	searchQueries = replaceAll(searchQueries, "=", "")
	searchQueries = replaceAll(searchQueries, "&", "")
	iframe.src = path.resolve(__dirname, `./pdfjsOriginal/web/viewer.html?file=${require('electron').remote.getGlobal('sharedObject').someProperty}#search=${searchQueries}`);
	viewerEle.appendChild(iframe);
	
	iframe.onload = function() {
		iframe.contentDocument.addEventListener('funcready', () => {
			let f = function(backward = false) {
				iframe.contentWindow.jumpToNextMatch(backward);
				$("#capeResult").empty().append(currArr[iframe.contentWindow.getCurrIndex()]);
				console.log(currArr[iframe.contentWindow.getCurrIndex()])
			}
			$('.answerarrow.arrowleft').off().click(() => f(true));
			$('.answerarrow.arrowright').off().click(() => f());
		});
	}	
}

function replaceAll(str, find, replace) {
    return str.replace(new RegExp(escapeRegExp(find), 'g'), replace);
}

function escapeRegExp(str) {
    return str.replace(/([.*+?^=!:${}()|\[\]\/\\])/g, "\\$1");
}

function getPDFText(firstPage, lastPage) {
	return new Promise(function(resolve, reject) {
		var key = PDF_URL.concat("textForEachPage").replace(".", "")
		console.log(key)
		console.log(store.has(key))
		if(store.has(key)) {
			var strings = ""
			var arrayTextByPage = store.get(key);
			for(var i = firstPage - 1; i <=  lastPage - 1; i++){
				strings = strings.concat(arrayTextByPage[i])
			}
			resolve(strings)
		} else {
			var gethtml = getHtml()	
			gethtml.then((data) => {
				console.log(data)
				store.set(key, data)
				console.log(store.store)
				var strings = ""
				for(var i = firstPage - 1; i <=  lastPage - 1; i++){
					strings = strings.concat(data[i])
				}
				resolve(strings)
			})
		}
		
	})
}

function getHtml() {
	return new Promise(function(resolve, reject) {
		var getlayered = getLayeredText()
		getlayered.then((data) => {
			var gettextaftermap = getTextAfterMap()
			gettextaftermap.then((data) => {
				console.log(data);
				resolve(data);
			})
		})
	})
}

function getLayeredText() {

	return new Promise(function(resolve, reject) {
		console.log("Inside of getLayeredText")
		map.clear()
		var pdfdoc = iframe.contentWindow.getPdfDocument()
		var lastPromise; // will be used to chain promises
		lastPromise = pdfdoc.getMetadata().then(function(data) {
		});

		var loadPage = function(pageNum) {
			return pdfdoc.getPage(pageNum).then(function(page) {
				return page.getTextContent().then(function(content) {
					var strings = content.items.map(function(item) {
						if(map.get(Math.round(item.height))) {
							map.set(Math.round(item.height), map.get(Math.round(item.height)) + item.str.length);
						} else {
							map.set(Math.round(item.height), item.str.length)
						}
						return item.str;
					});
				}).then(function() {
					console.log(pageNum)
					if(pageNum == pdfdoc.numPages) {
						resolve("DONE")
					}
				});
			});
		};
		for (var i = 1; i <= pdfdoc.numPages; i++) {
			lastPromise = lastPromise.then(loadPage.bind(null, i));
		}
	})
}

function getNumPages() {
	return iframe.contentWindow.getPdfDocument().numPages
}

function getTextAfterMap(){
	return new Promise(function(resolve, reject) {
		var textForEachPage = []
		var maxFont = 0
		var maxFontFreq = 0
		map.forEach((value, key) => {
			if(value > maxFontFreq){
				maxFontFreq = value;
				maxFont = key
			}
		})
		var pdfdoc = iframe.contentWindow.getPdfDocument()
		var lastPromise; // will be used to chain promises
		lastPromise = pdfdoc.getMetadata().then(function(data) {
		});

		var loadPage = function(pageNum) {
			return pdfdoc.getPage(pageNum).then(function(page) {
				var viewport = page.getViewport({
					scale: 1.0,
				});
				return page.getTextContent().then(function(content) {
					var strings = content.items.map(function(item) {
						if(Math.round(item.height) == maxFont) {
							return item.str;
						}
					});
					strings = strings.join(' ');
					textForEachPage.push(strings)
				}).then(function() {
					if(pdfdoc.numPages === pageNum) {
						resolve(textForEachPage)
					}
				});
			});
		};

		for (var i = 1; i <= pdfdoc.numPages; i++) {
			lastPromise = lastPromise.then(loadPage.bind(null, i));
		}
	})
}



Alertify = function () {

	var _alertify = {},
	    dialogs   = {},
	    isopen    = false,
	    keys      = { ENTER: 13, ESC: 27, SPACE: 32 },
	    queue     = [],
	    $, btnCancel, btnOK, btnReset, btnResetBack, btnFocus, elCallee, elCover, elDialog, elLog, form, input, getTransitionEvent;

	/**
	 * Markup pieces
	 * @type {Object}
	 */
	dialogs = {
		buttons : {
			holder : "<nav class=\"alertify-buttons\">{{buttons}}</nav>",
			submit : "<button type=\"submit\" class=\"alertify-button alertify-button-ok\" id=\"alertify-ok\">{{ok}}</button>",
			ok     : "<button class=\"alertify-button alertify-button-ok\" id=\"alertify-ok\">{{ok}}</button>",
			cancel : "<button class=\"alertify-button alertify-button-cancel\" id=\"alertify-cancel\">{{cancel}}</button>"
		},
		input   : "<div class=\"alertify-text-wrapper\"><input type=\"text\" class=\"alertify-text\" id=\"alertify-text\"></div>",
		message : "<p class=\"alertify-message\">{{message}}</p>",
		log     : "<article class=\"alertify-log{{class}}\">{{message}}</article>"
	};

	/**
	 * Return the proper transitionend event
	 * @return {String}    Transition type string
	 */
	getTransitionEvent = function () {
		var t,
		    type,
		    supported   = false,
		    el          = document.createElement("fakeelement"),
		    transitions = {
			    "WebkitTransition" : "webkitTransitionEnd",
			    "MozTransition"    : "transitionend",
			    "OTransition"      : "otransitionend",
			    "transition"       : "transitionend"
		    };

		for (t in transitions) {
			if (el.style[t] !== undefined) {
				type      = transitions[t];
				supported = true;
				break;
			}
		}

		return {
			type      : type,
			supported : supported
		};
	};

	/**
	 * Shorthand for document.getElementById()
	 *
	 * @param  {String} id    A specific element ID
	 * @return {Object}       HTML element
	 */
	$ = function (id) {
		return document.getElementById(id);
	};

	/**
	 * Alertify private object
	 * @type {Object}
	 */
	_alertify = {

		/**
		 * Labels object
		 * @type {Object}
		 */
		labels : {
			ok     : "OK",
			cancel : "Cancel"
		},

		/**
		 * Delay number
		 * @type {Number}
		 */
		delay : 5000,

		/**
		 * Whether buttons are reversed (default is secondary/primary)
		 * @type {Boolean}
		 */
		buttonReverse : false,

		/**
		 * Which button should be focused by default
		 * @type {String}	"ok" (default), "cancel", or "none"
		 */
		buttonFocus : "ok",

		/**
		 * Set the transition event on load
		 * @type {[type]}
		 */
		transition : undefined,

		/**
		 * Set the proper button click events
		 *
		 * @param {Function} fn    [Optional] Callback function
		 *
		 * @return {undefined}
		 */
		addListeners : function (fn) {
			var hasOK     = (typeof btnOK !== "undefined"),
			    hasCancel = (typeof btnCancel !== "undefined"),
			    hasInput  = (typeof input !== "undefined"),
			    val       = "",
			    self      = this,
			    ok, cancel, common, key, reset;

			// ok event handler
			ok = function (event) {
				if (typeof event.preventDefault !== "undefined") event.preventDefault();
				common(event);
				if (typeof input !== "undefined") val = input.value;
				if (typeof fn === "function") {
					if (typeof input !== "undefined") {
						fn(true, val);
					}
					else fn(true);
				}
				return false;
			};

			// cancel event handler
			cancel = function (event) {
				if (typeof event.preventDefault !== "undefined") event.preventDefault();
				common(event);
				if (typeof fn === "function") fn(false);
				return false;
			};

			// common event handler (keyup, ok and cancel)
			common = function (event) {
				self.hide();
				self.unbind(document.body, "keyup", key);
				self.unbind(btnReset, "focus", reset);
				if (hasOK) self.unbind(btnOK, "click", ok);
				if (hasCancel) self.unbind(btnCancel, "click", cancel);
			};

			// keyup handler
			key = function (event) {
				var keyCode = event.keyCode;
				if ((keyCode === keys.SPACE && !hasInput) || (hasInput && keyCode === keys.ENTER)) ok(event);
				if (keyCode === keys.ESC && hasCancel) cancel(event);
			};

			// reset focus to first item in the dialog
			reset = function (event) {
				if (hasInput) input.focus();
				else if (!hasCancel || self.buttonReverse) btnOK.focus();
				else btnCancel.focus();
			};

			// handle reset focus link
			// this ensures that the keyboard focus does not
			// ever leave the dialog box until an action has
			// been taken
			this.bind(btnReset, "focus", reset);
			this.bind(btnResetBack, "focus", reset);
			// handle OK click
			if (hasOK) this.bind(btnOK, "click", ok);
			// handle Cancel click
			if (hasCancel) this.bind(btnCancel, "click", cancel);
			// listen for keys, Cancel => ESC
			this.bind(document.body, "keyup", key);
			if (!this.transition.supported) {
				this.setFocus();
			}
		},

		/**
		 * Bind events to elements
		 *
		 * @param  {Object}   el       HTML Object
		 * @param  {Event}    event    Event to attach to element
		 * @param  {Function} fn       Callback function
		 *
		 * @return {undefined}
		 */
		bind : function (el, event, fn) {
			if (typeof el.addEventListener === "function") {
				el.addEventListener(event, fn, false);
			} else if (el.attachEvent) {
				el.attachEvent("on" + event, fn);
			}
		},

		/**
		 * Use alertify as the global error handler (using window.onerror)
		 *
		 * @return {boolean} success
		 */
		handleErrors : function () {
			if (typeof global.onerror !== "undefined") {
				var self = this;
				global.onerror = function (msg, url, line) {
					self.error("[" + msg + " on line " + line + " of " + url + "]", 0);
				};
				return true;
			} else {
				return false;
			}
		},

		/**
		 * Append button HTML strings
		 *
		 * @param {String} secondary    The secondary button HTML string
		 * @param {String} primary      The primary button HTML string
		 *
		 * @return {String}             The appended button HTML strings
		 */
		appendButtons : function (secondary, primary) {
			return this.buttonReverse ? primary + secondary : secondary + primary;
		},

		/**
		 * Build the proper message box
		 *
		 * @param  {Object} item    Current object in the queue
		 *
		 * @return {String}         An HTML string of the message box
		 */
		build : function (item) {
			var html    = "",
			    type    = item.type,
			    message = item.message,
			    css     = item.cssClass || "";

			html += "<div class=\"alertify-dialog\">";
			html += "<a id=\"alertify-resetFocusBack\" class=\"alertify-resetFocus\" href=\"#\">Reset Focus</a>";

			if (_alertify.buttonFocus === "none") html += "<a href=\"#\" id=\"alertify-noneFocus\" class=\"alertify-hidden\"></a>";

			// doens't require an actual form
			if (type === "prompt") html += "<div id=\"alertify-form\">";

			html += "<article class=\"alertify-inner\">";
			html += dialogs.message.replace("{{message}}", message);

			if (type === "prompt") html += dialogs.input;

			html += dialogs.buttons.holder;
			html += "</article>";

			if (type === "prompt") html += "</div>";

			html += "<a id=\"alertify-resetFocus\" class=\"alertify-resetFocus\" href=\"#\">Reset Focus</a>";
			html += "</div>";

			switch (type) {
			case "confirm":
				html = html.replace("{{buttons}}", this.appendButtons(dialogs.buttons.cancel, dialogs.buttons.ok));
				html = html.replace("{{ok}}", this.labels.ok).replace("{{cancel}}", this.labels.cancel);
				break;
			case "prompt":
				html = html.replace("{{buttons}}", this.appendButtons(dialogs.buttons.cancel, dialogs.buttons.submit));
				html = html.replace("{{ok}}", this.labels.ok).replace("{{cancel}}", this.labels.cancel);
				break;
			case "alert":
				html = html.replace("{{buttons}}", dialogs.buttons.ok);
				html = html.replace("{{ok}}", this.labels.ok);
				break;
			default:
				break;
			}

			elDialog.className = "alertify alertify-" + type + " " + css;
			elCover.className  = "alertify-cover";
			return html;
		},

		/**
		 * Close the log messages
		 *
		 * @param  {Object} elem    HTML Element of log message to close
		 * @param  {Number} wait    [optional] Time (in ms) to wait before automatically hiding the message, if 0 never hide
		 *
		 * @return {undefined}
		 */
		close : function (elem, wait) {
			// Unary Plus: +"2" === 2
			var timer = (wait && !isNaN(wait)) ? +wait : this.delay,
			    self  = this,
			    hideElement, transitionDone;

			// set click event on log messages
			this.bind(elem, "click", function () {
				hideElement(elem);
			});
			// Hide the dialog box after transition
			// This ensure it doens't block any element from being clicked
			transitionDone = function (event) {
				event.stopPropagation();
				// unbind event so function only gets called once
				self.unbind(this, self.transition.type, transitionDone);
				// remove log message
				elLog.removeChild(this);
				if (!elLog.hasChildNodes()) elLog.className += " alertify-logs-hidden";
			};
			// this sets the hide class to transition out
			// or removes the child if css transitions aren't supported
			hideElement = function (el) {
				// ensure element exists
				if (typeof el !== "undefined" && el.parentNode === elLog) {
					// whether CSS transition exists
					if (self.transition.supported) {
						self.bind(el, self.transition.type, transitionDone);
						el.className += " alertify-log-hide";
					} else {
						elLog.removeChild(el);
						if (!elLog.hasChildNodes()) elLog.className += " alertify-logs-hidden";
					}
				}
			};
			// never close (until click) if wait is set to 0
			if (wait === 0) return;
			// set timeout to auto close the log message
			setTimeout(function () { hideElement(elem); }, timer);
		},

		/**
		 * Create a dialog box
		 *
		 * @param  {String}   message        The message passed from the callee
		 * @param  {String}   type           Type of dialog to create
		 * @param  {Function} fn             [Optional] Callback function
		 * @param  {String}   placeholder    [Optional] Default value for prompt input field
		 * @param  {String}   cssClass       [Optional] Class(es) to append to dialog box
		 *
		 * @return {Object}
		 */
		dialog : function (message, type, fn, placeholder, cssClass) {
			// set the current active element
			// this allows the keyboard focus to be resetted
			// after the dialog box is closed
			elCallee = document.activeElement;
			// check to ensure the alertify dialog element
			// has been successfully created
			var check = function () {
				if ((elLog && elLog.scrollTop !== null) && (elCover && elCover.scrollTop !== null)) return;
				else check();
			};
			// error catching
			if (typeof message !== "string") throw new Error("message must be a string");
			if (typeof type !== "string") throw new Error("type must be a string");
			if (typeof fn !== "undefined" && typeof fn !== "function") throw new Error("fn must be a function");
			// initialize alertify if it hasn't already been done
			this.init();
			check();

			queue.push({ type: type, message: message, callback: fn, placeholder: placeholder, cssClass: cssClass });
			if (!isopen) this.setup();

			return this;
		},

		/**
		 * Extend the log method to create custom methods
		 *
		 * @param  {String} type    Custom method name
		 *
		 * @return {Function}
		 */
		extend : function (type) {
			if (typeof type !== "string") throw new Error("extend method must have exactly one paramter");
			return function (message, wait) {
				this.log(message, type, wait);
				return this;
			};
		},

		/**
		 * Hide the dialog and rest to defaults
		 *
		 * @return {undefined}
		 */
		hide : function () {
			var transitionDone,
			    self = this;
			// remove reference from queue
			queue.splice(0,1);
			// if items remaining in the queue
			if (queue.length > 0) this.setup(true);
			else {
				isopen = false;
				// Hide the dialog box after transition
				// This ensure it doens't block any element from being clicked
				transitionDone = function (event) {
					event.stopPropagation();
					// unbind event so function only gets called once
					self.unbind(elDialog, self.transition.type, transitionDone);
				};
				// whether CSS transition exists
				if (this.transition.supported) {
					this.bind(elDialog, this.transition.type, transitionDone);
					elDialog.className = "alertify alertify-hide alertify-hidden";
				} else {
					elDialog.className = "alertify alertify-hide alertify-hidden alertify-isHidden";
				}
				elCover.className  = "alertify-cover alertify-cover-hidden";
				// set focus to the last element or body
				// after the dialog is closed
				elCallee.focus();
			}
		},

		/**
		 * Initialize Alertify
		 * Create the 2 main elements
		 *
		 * @return {undefined}
		 */
		init : function () {
			// ensure legacy browsers support html5 tags
			document.createElement("nav");
			document.createElement("article");
			document.createElement("section");
			// cover
			if ($("alertify-cover") == null) {
				elCover = document.createElement("div");
				elCover.setAttribute("id", "alertify-cover");
				elCover.className = "alertify-cover alertify-cover-hidden";
				document.body.appendChild(elCover);
			}
			// main element
			if ($("alertify") == null) {
				isopen = false;
				queue = [];
				elDialog = document.createElement("section");
				elDialog.setAttribute("id", "alertify");
				elDialog.className = "alertify alertify-hidden";
				document.body.appendChild(elDialog);
			}
			// log element
			if ($("alertify-logs") == null) {
				elLog = document.createElement("section");
				elLog.setAttribute("id", "alertify-logs");
				elLog.className = "alertify-logs alertify-logs-hidden";
				document.body.appendChild(elLog);
			}
			// set tabindex attribute on body element
			// this allows script to give it focus
			// after the dialog is closed
			document.body.setAttribute("tabindex", "0");
			// set transition type
			this.transition = getTransitionEvent();
		},

		/**
		 * Show a new log message box
		 *
		 * @param  {String} message    The message passed from the callee
		 * @param  {String} type       [Optional] Optional type of log message
		 * @param  {Number} wait       [Optional] Time (in ms) to wait before auto-hiding the log
		 *
		 * @return {Object}
		 */
		log : function (message, type, wait) {
			// check to ensure the alertify dialog element
			// has been successfully created
			var check = function () {
				if (elLog && elLog.scrollTop !== null) return;
				else check();
			};
			// initialize alertify if it hasn't already been done
			this.init();
			check();

			elLog.className = "alertify-logs";
			this.notify(message, type, wait);
			return this;
		},

		/**
		 * Add new log message
		 * If a type is passed, a class name "alertify-log-{type}" will get added.
		 * This allows for custom look and feel for various types of notifications.
		 *
		 * @param  {String} message    The message passed from the callee
		 * @param  {String} type       [Optional] Type of log message
		 * @param  {Number} wait       [Optional] Time (in ms) to wait before auto-hiding
		 *
		 * @return {undefined}
		 */
		notify : function (message, type, wait) {
			var log = document.createElement("article");
			log.className = "alertify-log" + ((typeof type === "string" && type !== "") ? " alertify-log-" + type : "");
			log.innerHTML = message;
			// append child
			elLog.appendChild(log);
			// triggers the CSS animation
			setTimeout(function() { log.className = log.className + " alertify-log-show"; }, 50);
			this.close(log, wait);
		},

		/**
		 * Set properties
		 *
		 * @param {Object} args     Passing parameters
		 *
		 * @return {undefined}
		 */
		set : function (args) {
			var k;
			// error catching
			if (typeof args !== "object" && args instanceof Array) throw new Error("args must be an object");
			// set parameters
			for (k in args) {
				if (args.hasOwnProperty(k)) {
					this[k] = args[k];
				}
			}
		},

		/**
		 * Common place to set focus to proper element
		 *
		 * @return {undefined}
		 */
		setFocus : function () {
			if (input) {
				input.focus();
				input.select();
			}
			else btnFocus.focus();
		},

		/**
		 * Initiate all the required pieces for the dialog box
		 *
		 * @return {undefined}
		 */
		setup : function (fromQueue) {
			var item = queue[0],
			    self = this,
			    transitionDone;

			// dialog is open
			isopen = true;
			// Set button focus after transition
			transitionDone = function (event) {
				event.stopPropagation();
				self.setFocus();
				// unbind event so function only gets called once
				self.unbind(elDialog, self.transition.type, transitionDone);
			};
			// whether CSS transition exists
			if (this.transition.supported && !fromQueue) {
				this.bind(elDialog, this.transition.type, transitionDone);
			}
			// build the proper dialog HTML
			elDialog.innerHTML = this.build(item);
			// assign all the common elements
			btnReset  = $("alertify-resetFocus");
			btnResetBack  = $("alertify-resetFocusBack");
			btnOK     = $("alertify-ok")     || undefined;
			btnCancel = $("alertify-cancel") || undefined;
			btnFocus  = (_alertify.buttonFocus === "cancel") ? btnCancel : ((_alertify.buttonFocus === "none") ? $("alertify-noneFocus") : btnOK),
			input     = $("alertify-text")   || undefined;
			form      = $("alertify-form")   || undefined;
			// add placeholder value to the input field
			if (typeof item.placeholder === "string" && item.placeholder !== "") input.value = item.placeholder;
			if (fromQueue) this.setFocus();
			this.addListeners(item.callback);
		},

		/**
		 * Unbind events to elements
		 *
		 * @param  {Object}   el       HTML Object
		 * @param  {Event}    event    Event to detach to element
		 * @param  {Function} fn       Callback function
		 *
		 * @return {undefined}
		 */
		unbind : function (el, event, fn) {
			if (typeof el.removeEventListener === "function") {
				el.removeEventListener(event, fn, false);
			} else if (el.detachEvent) {
				el.detachEvent("on" + event, fn);
			}
		}
	};

	return {
		alert   : function (message, fn, cssClass) { _alertify.dialog(message, "alert", fn, "", cssClass); return this; },
		confirm : function (message, fn, cssClass) { _alertify.dialog(message, "confirm", fn, "", cssClass); return this; },
		extend  : _alertify.extend,
		init    : _alertify.init,
		log     : function (message, type, wait) { _alertify.log(message, type, wait); return this; },
		prompt  : function (message, fn, placeholder, cssClass) { _alertify.dialog(message, "prompt", fn, placeholder, cssClass); return this; },
		success : function (message, wait) { _alertify.log(message, "success", wait); return this; },
		error   : function (message, wait) { _alertify.log(message, "error", wait); return this; },
		set     : function (args) { _alertify.set(args); },
		labels  : _alertify.labels,
		debug   : _alertify.handleErrors
	};
};

// AMD and window support
if (typeof define === "function") {
	define([], function () { return new Alertify(); });
} else if (typeof global.alertify === "undefined") {
	global.alertify = new Alertify();
}

$('button.success').click(function() {
	console.log('clicked!')
  alertify.set({ delay: 1700 });
  							alertify.success("Success notification");  
});

$('button.alert').click(function() {
	console.log('clicked!')
    alertify.set({ delay: 1700 });
	    							alertify.error("Error notification");  
});


 // Smooth scroll for the navigation and links with .scrollto classes
 $('.main-nav a, .mobile-nav a, .scrollto').on('click', function() {
    if (location.pathname.replace(/^\//, '') == this.pathname.replace(/^\//, '') && location.hostname == this.hostname) {
      var target = $(this.hash);
      if (target.length) {
        var top_space = 0;

        if ($('#header').length) {
          top_space = $('#header').outerHeight();

          if (! $('#header').hasClass('header-scrolled')) {
            top_space = top_space - 40;
          }
        }

        $('html, body').animate({
          scrollTop: target.offset().top - top_space
        }, 1500, 'easeInOutExpo');

        if ($(this).parents('.main-nav, .mobile-nav').length) {
          $('.main-nav .active, .mobile-nav .active').removeClass('active');
          $(this).closest('li').addClass('active');
        }

        if ($('body').hasClass('mobile-nav-active')) {
          $('body').removeClass('mobile-nav-active');
          $('.mobile-nav-toggle i').toggleClass('fa-times fa-bars');
          $('.mobile-nav-overly').fadeOut();
        }
        return false;
      }
    }
  });

  (function ($) {
    "use strict";
  
    // Mobile Navigation
    console.log($('.main-nav').length)
    if ($('.main-nav').length) {
      var $mobile_nav = $('.main-nav').clone().prop({
        class: 'mobile-nav d-lg-none'
      });
      $('body').append($mobile_nav);
      $('body').prepend('<button type="button" class="mobile-nav-toggle d-lg-none"><i class="fa fa-bars"></i></button>');
      $('body').append('<div class="mobile-nav-overly"></div>');
  
      $(document).on('click', '.mobile-nav-toggle', function(e) {
        $('body').toggleClass('mobile-nav-active');
        $('.mobile-nav-toggle i').toggleClass('fa-times fa-bars');
        $('.mobile-nav-overly').toggle();
      });
      
      $(document).on('click', '.mobile-nav .drop-down > a', function(e) {
        e.preventDefault();
        $(this).next().slideToggle(300);
        $(this).parent().toggleClass('active');
      });
  
      $(document).click(function(e) {
        var container = $(".mobile-nav, .mobile-nav-toggle");
        if (!container.is(e.target) && container.has(e.target).length === 0) {
          if ($('body').hasClass('mobile-nav-active')) {
            $('body').removeClass('mobile-nav-active');
            $('.mobile-nav-toggle i').toggleClass('fa-times fa-bars');
            $('.mobile-nav-overly').fadeOut();
          }
        }
      });
    } else if ($(".mobile-nav, .mobile-nav-toggle").length) {
      $(".mobile-nav, .mobile-nav-toggle").hide();
    }
  
  })(jQuery);

  