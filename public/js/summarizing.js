const { dialog } = require('electron').remote;
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

var tools = require('./createFile/coordinates.js')
var bookmarkArray = [];
var Tokenizer = require('sentence-tokenizer');
var tokenizer = new Tokenizer('Chuck');
const viewerEle = document.getElementById('viewer');
viewerEle.innerHTML = ''; // destroy the old instance of PDF.js (if it exists)
const iframe = document.createElement('iframe');
iframe.src = path.resolve(__dirname, `./pdfjsOriginal/web/viewer.html?file=${require('electron').remote.getGlobal('sharedObject').someProperty}`);

const etudeFilepath = __dirname.replace("/public/js","").replace("\\public\\js","")
const secVersionFilepath = etudeFilepath + "/folderForHighlightedPDF/secVersion.pdf"

viewerEle.appendChild(iframe);

filepath = require('electron').remote.getGlobal('sharedObject').someProperty;
deepai.setApiKey('a5c8170e-046a-4c56-acb1-27c37049b193');
//get text from pdf to send to flask backends
var PDF_URL  = filepath;
var capeClicked = false;
var btnClicked = false;
var bookmarkOpened = false;
var java = require('java');
if(osvers == "darwin"){
	java.classpath.push(etudeFilepath + "/MacKernel.jar");
	java.classpath.push(etudeFilepath + "/Contents/Resources/Wolfram\ Player.app/Contents/SystemFiles/Links/JLink/JLink.jar");
} else {
	java.classpath.push(etudeFilepath + "/WindowsKernel.jar");
	java.classpath.push(etudeFilepath + "/12.0/SystemFiles/Links/JLink/JLink.jar");
}
//console.log(output.findTextAnswerSync('foo','bar', 1, "Sentence"));
//njava.classpath.push("/Applications/Wolfram\ Desktop.app/Contents/SystemFiles/Links/JLink/JLink.jar")
var kernel = java.newInstanceSync('p1.Kernel', etudeFilepath);
var htmlForEntireDoc = ""
pdfAllToHTML(PDF_URL);
var numPages = 0;
PDFJS.getDocument({ url: PDF_URL }).then(function(pdf_doc) {
  __PDF_DOC = pdf_doc;
  numPages = __PDF_DOC.numPages;
});


$("#bookmark_icon").click(function(){
  //get the page number
  var whichpagetobookmark = document.getElementsByTagName('iframe')[0].contentWindow.document.getElementById('pageNumber').value;
  for (var j = 0; j < bookmarkArray.length; j++){
	if (bookmarkArray[j] == whichpagetobookmark)
	  return;
  }
  show_nextItem(whichpagetobookmark,null);
  //store the page number in the database
  bookmarkArray.push(whichpagetobookmark);
  showPDF(filepath,parseInt(whichpagetobookmark));
})

function showPDF(pdf_url,bookmark_page) {
  //PDFJS.GlobalWorkerOptions.workerSrc ='../../node_modules/pdfjs-dist/build/pdf.worker.js';
	PDFJS.getDocument({ url: pdf_url }).then(function(pdf_doc) {
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
						var __CANVAS = $('.bookmark-canvas').get($(".bookmark-canvas").length-1),
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

function show_nextItem(whichpage, removeWhich)
		{
				var next_text = "<div class='bookmark_section col-md-2 col-lg-2' style = 'margin-top:10px;'><div><canvas class='bookmark-canvas' data = '"+whichpage+"'></canvas><img class = 'deleteImage_' src='public/images/bookmarkminus.png' data = '"+removeWhich+"' id = 'myButton'/></div></div>";
				var next_div = document.createElement("div");
				next_div.innerHTML = next_text;
				document.getElementById('bookmark_container').append(next_div);
		}
  // delete the bookmark in the bookmark page
$(document).on("click",".deleteImage_", function(){
				($(this).parent()).parent().remove();

});
// when the user click the bookmark
$(document).on("click",".bookmark-canvas", function(){
	console.log(document.getElementsByTagName('iframe')[0].contentWindow.document.getElementById('thumbnailView').children)
	console.log(document.getElementsByTagName('iframe')[0].contentWindow.document.getElementById('thumbnailView').children[$(this).attr("data") - 1])
  	document.getElementsByTagName('iframe')[0].contentWindow.document.getElementById('thumbnailView').children[$(this).attr("data") - 1].click()
});


$("#cape_btn").click(function(){
	if(capeClicked){
		document.getElementById("myDropdown").classList.toggle("show");
	}
	setTimeout(function(){ 
		if(htmlForEntireDoc == ""){
			htmlForEntireDoc = htmlWholeFileToPartialPlainText(1, numPages);
		}
		htmlForEntireDoc.then((x) => {
			var promiseToAppend = new Promise(function(resolve, reject){
				var answer = kernel.findTextAnswerSync(x, $("#questionVal").val(), 2, "Sentence");
				updateHighlights(answer, true)
				$("#capeResult").empty().append(answer);
				resolve("GOOD")
			});
			//$("#capeResult").empty().append(kernel.findTextAnswerSync(x, $("#questionVal").val(), 2, "Sentence"));
			//document.getElementById("myDropdown").classList.toggle("show");
			promiseToAppend.then((data) => {
				document.getElementById("myDropdown").classList.toggle("show");
			});
		});
		capeClicked = true;
	}, 1);
});

$("#help").click(function(){
  // close the bookmark page
  $("#bookmark_item").attr("data","true");
  $("#bookmark_item").click();

  $('.help_popup').show();
	$('.help_popup').click(function(){
	$('.help_popup').hide();
  });
})

//summarization function
$('#summarizingButton').click(function(){
  $('.su_popup').hide();
  summaryButtonPressed($('#pageRange').val(),$('#topageRange').val());
  // here you can add the loading button
  $('.summarizer_loading').show();
  // $('.hover_bkgr_fricc').click(function(){
  //       $('.hover_bkgr_fricc').hide();
  //   });
})

$('#escapeSUPopupButton').click(function(){
	$('.su_popup').hide();
})

var textDsum = "";
var iPagesum = 0;
var iEndPagesum = 0;
function processSummarizationResult(t){
  noLineBreakText = t["output"].replace(/(\r\n|\n|\r)/gm, " ");

  tokenizer.setEntry(noLineBreakText);

  updateHighlights(tokenizer.getSentences(), false)
  $("#summarizingResult").empty().append(t["output"]);
  //here you can remove the loading button
  $('.summarizer_loading').hide();
  // $('.hover_bkgr_fricc').show();
  iPagesum = 0;
  iEndPagesum = 0;
  textDsum = 0;
};

function pdfAllToHTML(nameOfFileDir) {
  var exec = require('child_process').exec, child;

  var filenamewithextension = path.parse(nameOfFileDir).base;
  filenamewithextension = filenamewithextension.split('.')[0];
  console.log(filenamewithextension)
  //update directory to JAR file
  var pathOfFile = etudeFilepath + '/tmp/' + filenamewithextension + '.html'
  try {
	if (fs.existsSync(pathOfFile)) {
	  console.log("html exists already")
	  return;
	}
  } catch(err) {
	console.error(err)
  }
  let executionstring = 'java -jar ' + etudeFilepath + '/PDFToHTML.jar \"' + nameOfFileDir + '\" \"' + etudeFilepath +  '/tmp/' + filenamewithextension + '.html\"';
  child = exec(executionstring,
	  function (error, stdout, stderr) {
		  if (error !== null) {
			   console.log('exec error: ' + error);
		  }
	  });
}
function summaryButtonPressed(firstpage, lastpage) {
  var htmlStuff = htmlWholeFileToPartialPlainText(firstpage, lastpage);
  htmlStuff.then((x) => {
	deepai.callStandardApi("summarization", {text: x}).then((resp) => processSummarizationResult(resp))
  });
}

function htmlWholeFileToPartialPlainText(firstpage, lastpage) {
  return new Promise(function(resolve, reject){
	var filenamewithextension = path.parse(PDF_URL).base;
	filenamewithextension = filenamewithextension.split('.')[0];
	var outputfile = etudeFilepath + '/tmp/' + filenamewithextension + '.html';
	const htmlToJson = require('html-to-json');
	let bigarray = [];
	let bigarrayback = [];
	//the correct html file directory within our project
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
		let index = (item.substring(0,item.search("pt")));
		function checkSize(age) {
		  return age == index;
		}
		if(bigarrayback.findIndex(checkSize) == -1) {
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

$('#getRangeButton').click(function(){
  //close the bookmark page
  $("#bookmark_item").attr("data","true");
  $("#bookmark_item").click();

  //$('#getRangeButton').hide();
  $('.su_popup').show();
})

// kernel.findTextAnswerSync('foo','bar', 1, "Sentence");
// console.log('hello');
function updateHighlights(sentences, isSearch){
	if(fs.existsSync(etudeFilepath + '/folderForHighlightedPDF/secVersion.pdf')){
		fs.unlinkSync(etudeFilepath + "/folderForHighlightedPDF/secVersion.pdf");
  	}
  tools.extractor(sentences,filepath, etudeFilepath + '/folderForHighlightedPDF/secVersion.pdf', etudeFilepath);
  checkFlag(isSearch);
}
function checkForJump(){
	if(!fs.existsSync(etudeFilepath + "/tmp/object.json")){
		window.setTimeout(checkForJump, 100);
	} else {
		var contents = fs.readFileSync(etudeFilepath + "/tmp/object.json");
		var jsonContents = JSON.parse(contents)
		console.log(jsonContents)
		jumpPage(jsonContents[0]['page'] + 1);
	}
}

function jumpPage(pageNumber){
	if(document.getElementsByTagName('iframe')[0].contentWindow.document.getElementById('thumbnailView') != null && document.getElementsByTagName('iframe')[0].contentWindow.document.getElementById('thumbnailView').childNodes[pageNumber - 1] != undefined){
		document.getElementsByTagName('iframe')[0].contentWindow.document.getElementById('thumbnailView').childNodes[pageNumber - 1].click()
	} else {
		window.setTimeout(jumpPage, 0, pageNumber)
	}
}
function checkFlag(isSearch) {
	if(!fs.existsSync(etudeFilepath + '/folderForHighlightedPDF/secVersion.pdf')){
	  console.log("checking")
	  window.setTimeout(checkFlag, 100, isSearch); /* this checks the flag every 100 milliseconds*/
	} else {
	  viewerEle.innerHTML = "";
	  iframe.src = path.resolve(__dirname, `./pdfjsOriginal/web/viewer.html?file=${secVersionFilepath}`);
	  console.log(iframe)
	  viewerEle.appendChild(iframe);
	  console.log("DONE")
	  console.log("Whether you want to search: " + isSearch)
	  if(isSearch){
	  	console.log("CHECKINGFORJUMP")
	  	checkForJump()
	  }
	}
}

