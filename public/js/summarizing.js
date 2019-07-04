const { dialog } = require('electron').remote;
const path = require('path');

const fs = require('fs');
var mkdirp = require('mkdirp'),
  request = require('request'),
  FormData = require('form-data'),
  async = require('async');

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
//get text from pdf to send to flask backends
var PDF_URL  = filepath;
console.log(PDF_URL);
var capeClicked = false;
var btnClicked = false;
var bookmarkOpened = false;
var java = require('java');
//java.classpath.push("./Kernel.jar");
//java.classpath.push("./Contents/Resources/Wolfram Player.app/Contents/SystemFiles/Links/JLink/JLink.jar");
//njava.classpath.push("/Applications/Wolfram\ Desktop.app/Contents/SystemFiles/Links/JLink/JLink.jar")
//var kernel = java.newInstanceSync('p1.Kernel');

$("#bookmark_icon").click(function(){
  //get the page number
  var whichpagetobookmark = document.getElementsByTagName('iframe')[0].contentWindow.document.getElementById('pageNumber').value;
  console.log(bookmarkArray.length);
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
  console.log($(this).attr("data"));
  console.log("above you clicked sth");
  document.getElementsByTagName('iframe')[0].contentWindow.document.getElementById('thumbnailView').children[$(this).attr("data") - 1].click()
});


$("#cape_btn").click(function(){
  //get the text and question
  // document.getElementById("myDropdown").classList.toggle("show");
  if (capeClicked == true)
  {
    document.getElementById("myDropdown").classList.toggle("show");
    capeClicked = false;
  }
  else{
    getSeletedPageText($('#pageRange').val(),$('#topageRange').val());
    capeClicked = true;
  }

})

function getPageText(pageNum, PDFDocumentInstance) {
  // Return a Promise that is solved once the text of the page is retrieven
  return new Promise(function (resolve, reject) {
      PDFDocumentInstance.getPage(pageNum).then(function (pdfPage) {
          // The main trick to obtain the text of the PDF page, use the getTextContent method
          pdfPage.getTextContent().then(function (textContent) {
              var textItems = textContent.items;
              var finalString = "";

              // Concatenate the string of the item to the final string
              for (var i = 0; i < textItems.length; i++) {
                  var item = textItems[i];

                  finalString += item.str + " ";
              }

              // Solve promise with the text retrieven from the page
              resolve(finalString);
          });
      });
  });
}


$("#help").click(function(){
  // close the bookmark page
  $("#bookmark_item").attr("data","true");
  $("#bookmark_item").click();

  $('.help_popup').show();
	$('.help_popup').click(function(){
    $('.help_popup').hide();
  });
})

//get text function
var textD = "";
var iPage = 0;
var iEndPage = 0;
function getSeletedPageText(fpage,tpage)
{
  PDFJS.getDocument(PDF_URL).then(function (PDFDocumentInstance) {

      iPage = parseInt(fpage);
      iEndPage = parseInt(tpage);
      getTextByPage(PDFDocumentInstance);
    }, function (reason) {
        // PDF loading error
        console.error(reason);
        console.log("eorror");
    });
}


function getTextByPage(instance){
  getPageText(iPage , instance).then(function(textPage){
    if(iPage != 0)
      textD += textPage
    if(iPage < iEndPage){
      iPage++;
      getTextByPage(instance)
    }else{
      console.log("BANANA");
      console.log("succeeded");
      //$("#capeResult").empty().append(kernel.findTextAnswerSync(textD, $("#questionVal").val(), 2, "Sentence"));
      // $('.hover_bkgr_fricc').show();
      document.getElementById("myDropdown").classList.toggle("show");
       //init for search
      iPage = 0;
      iEndPage = 0;
      textD = 0;
      capeClicked = true;
      //post text.
      return;
    }
  });
}

//summarization function
$('#summarizingButton').click(function(){
  console.log("summarizingButtonClicked");
  $('.su_popup').hide();
  getSeletedPageTextForSummarization($('#pageRange').val(),$('#topageRange').val());

  // here you can add the loading button
  $('.summarizer_loading').show();
  $('.hover_bkgr_fricc').click(function(){
        $('.hover_bkgr_fricc').hide();
    });
})

var textDsum = "";
var iPagesum = 0;
var iEndPagesum = 0;
function getSeletedPageTextForSummarization(fpage,tpage)
{
  PDFJS.getDocument(PDF_URL).then(function (PDFDocumentInstanceSummarizing) {
      iPagesum = parseInt(fpage);
      iEndPagesum = parseInt(tpage);
      getTextByPageForSummarization(PDFDocumentInstanceSummarizing);
    }, function (reason) {
        // PDF loading error
        console.error(reason);
        console.log("eorror");
    });
}

function processSummarizationResult(t){
  console.log("succeeded");
  console.log(t);
  console.log(typeof(t));
  console.log(t["output"])
  $("#summarizingResult").empty().append(t["output"]);
  //here you can remove the loading button
  $('.summarizer_loading').hide();
  $('.hover_bkgr_fricc').show();
  iPagesum = 0;
  iEndPagesum = 0;
  textDsum = 0;
};

function getTextByPageForSummarization(instance){
  getPageText(iPagesum , instance).then(function(textPage){
    if(iPagesum != 0)
      textDsum += textPage
    if(iPagesum < iEndPagesum){
      iPagesum++;
      getTextByPageForSummarization(instance)
    }else{
      // viewerEle.innerHTML = "";
      // iframe.src = path.resolve(__dirname, `./pdfjsOriginal/web/viewer.html?file=${'/Users/etashguha/Downloads/Sparrow2.pdf'}`);
      // viewerEle.appendChild(iframe);
      console.log(textDsum)
      deepai.setApiKey('a5c8170e-046a-4c56-acb1-27c37049b193');
      deepai.callStandardApi("summarization", {
        text: textDsum}).then((resp) => {
          if(fs.existsSync('./folderForHighlightedPDF/secVersion.pdf')){
            fs.unlinkSync("./folderForHighlightedPDF/secVersion.pdf");
          }
          processSummarizationResult(resp)
          noLineBreakText = resp["output"].replace(/(\r\n|\n|\r)/gm, " ");

          tokenizer.setEntry(noLineBreakText);
          console.log(tokenizer.getSentences());

          tools.extractor(tokenizer.getSentences(),filepath, './folderForHighlightedPDF/secVersion.pdf');
          checkFlag()
          
        });
      return;
    }
  });
}

$('#getRangeButton').click(function(){
  //close the bookmark page
  $("#bookmark_item").attr("data","true");
  $("#bookmark_item").click();

  $('#getRangeButton').hide();
  $('.su_popup').show();
})

//kernel.findTextAnswerSync('foo','bar', 1, "Sentence");
console.log('hello');

function checkFlag() {
    if(!fs.existsSync('./folderForHighlightedPDF/secVersion.pdf')){
      console.log("checking")
      window.setTimeout(checkFlag, 100); /* this checks the flag every 100 milliseconds*/
    } else {
      viewerEle.innerHTML = "";
      iframe.src = path.resolve(__dirname, `./pdfjsOriginal/web/viewer.html?file=${secVersionFilepath}`);
      console.log(iframe)
      viewerEle.appendChild(iframe);
      console.log("DONE")
    }
}

