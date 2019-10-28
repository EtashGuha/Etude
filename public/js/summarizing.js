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

var closeButton = document.getElementById("closeButton");
closeButton.addEventListener('click', () => {
    win.close();
})

var maxbuttton = document.getElementById("maxbutton");
maxbuttton.addEventListener('click', () => {
    win.maximize();
})

var minbutton = document.getElementById("minbutton");
minbutton.addEventListener('click', () => {
    win.minimize();
})



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
	if (capeClicked) {
		document.getElementById("myDropdown").classList.toggle("show");
	}
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
							map.set(Math.round(item.height), map.get(Math.round(item.height)) + 1);
						} else {
							map.set(Math.round(item.height), 1)
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

  