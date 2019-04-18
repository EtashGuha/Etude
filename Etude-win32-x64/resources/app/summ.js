const { dialog } = require('electron').remote;
const path = require('path');
var textData = null;

// Add an event listener to our button.
// document.getElementById('myButton').addEventListener('click', () => {

  // When the button is clicked, open the native file picker to select a PDF.
  // dialog.showOpenDialog({
  //   properties: ['openFile'], // set to use openFileDialog
  //   filters: [ { name: "PDFs", extensions: ['pdf'] } ] // limit the picker to just pdfs
  // }, (filepaths) => {

    // Since we only allow one file, just use the first one
    // const filePath = filepaths[0];

const viewerEle = document.getElementById('viewer');
viewerEle.innerHTML = ''; // destroy the old instance of PDF.js (if it exists)

// create an instance of WV.
const viewer = new PDFTron.WebViewer({
  path: './public/WebViewer/lib',
  l: 'demo:hello@xecure.systems:75c753bd01944dfda6ffc7abdac5421ddeeb98bc25142d2039',
  initialDoc: require('electron').remote.getGlobal('sharedObject').someProperty,
  css: './public/css/pdf.css'
}, viewerEle)

// var input = document.getElementById('file_upload');
viewerEle.addEventListener('ready', function() {
  var viewerInstance = viewer.getInstance(); // instance is ready here
  // input.addEventListener('change', function() {
  //   // var file = input.files[0]; 
  //   // viewerInstance.loadDocument(file, { filename: file.name })
  
    console.log("insideinsdieinsideinsideinside");
    console.log(viewerInstance.getPageCount());
  $('#toPage').empty().append(viewerInstance.getPageCount());
    $('#gotoNextPage').click(function(){
        viewerInstance.goToNextPage();
        $('#cuPage').empty().append(viewerInstance.getCurrentPageNumber());
    });
    $('#gotoPreviousPage').click(function(){
        viewerInstance.goToPrevPage();
        $('#cuPage').empty().append(viewerInstance.getCurrentPageNumber());
    });
    // getTotalNum(viewerInstance.getPageCount());
    // $('#gotoPreviousPage').click();
  // });
});

filepath = require('electron').remote.getGlobal('sharedObject').someProperty;
   //get text from pdf to send to flask backends
var PDF_URL  = filepath;
console.log(PDF_URL);
console.log("this is the pdfurl");
PDFJS.getDocument(PDF_URL).then(function (PDFDocumentInstance) {
 
     var totalPages = PDFDocumentInstance.pdfInfo.numPages;
     var pageNumber = $('#pageRange').val();
     var topageNumber = $('#topageRange').val();
     // Extract the text
     for (var i = parseInt(pageNumber); i <=(parseInt(pageNumber)+parseInt(topageNumber)); i++){
        getPageText(i , PDFDocumentInstance).then(function(textPage){
            // Show the text of the page in the console
            textData += textPage;
            console.log(textData);
            console.log("this is text you see");
        });
     }
     
     
 
 }, function (reason) {
     // PDF loading error
     console.error(reason);
 });  
    

$("#cape_btn").click(function(){

  PDFJS.getDocument(PDF_URL).then(function (PDFDocumentInstance) {
    textData = null;
    var totalPages = PDFDocumentInstance.pdfInfo.numPages;
    var pageNumber = $('#pageRange').val();
    var topageNumber = $('#topageRange').val();
    console.log("dfdfkhdakhihihihihi");
    console.log(pageNumber);
    // Extract the text
    for (var i = parseInt(pageNumber); i <=(parseInt(pageNumber)+parseInt(topageNumber)); i++){
       getPageText(i , PDFDocumentInstance).then(function(textPage){
           // Show the text of the page in the console
           textData += textPage;
       });
    }
    
    

    }, function (reason) {
        // PDF loading error
        console.error(reason);
    });  

  var quest = $("#questionVal").val();
  console.log(quest);
  // console.log(textData);
  $.ajax({
    url:"http://52.53.162.33:5000/cape",
    data: {
      pdfData: textData,
      question: quest
    },
    method: "POST",
    // dataType: "json"
  }).done(function(t){
    console.log("succeeded");
    console.log(t);
    $("#capeResult").empty().append(t);
    // $('.hover_bkgr_fricc').show();
    document.getElementById("myDropdown").classList.toggle("show");
  })
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

// cape api show
function cape(){
	$('.hover_bkgr_fricc').show();
	$('.hover_bkgr_fricc').click(function(){
        $('.hover_bkgr_fricc').hide();
    });
    $('.popupCloseButton').click(function(){
        $('.hover_bkgr_fricc').hide();
    });
  
}
$('#summarizingButton').click(function(){
  // textData = null;
  PDFJS.getDocument(PDF_URL).then(function (PDFDocumentInstance) {
    textData = null;
    var totalPages = PDFDocumentInstance.pdfInfo.numPages;
    var pageNumber = $('#pageRange').val();
    var topageNumber = $('#topageRange').val();
    console.log("dfdfkhdakhihihihihi");
    console.log(pageNumber);
    // Extract the text
    for (var i = parseInt(pageNumber); i <=(parseInt(pageNumber)+parseInt(topageNumber)); i++){
       getPageText(i , PDFDocumentInstance).then(function(textPage){
           // Show the text of the page in the console
           textData += textPage;
       });
    }
    
    

}, function (reason) {
    // PDF loading error
    console.error(reason);
});  
  // $('.hover_bkgr_fricc').show();
	$('.hover_bkgr_fricc').click(function(){
        $('.hover_bkgr_fricc').hide();
    });
    $('.popupCloseButton').click(function(){
        $('.hover_bkgr_fricc').hide();
    });
    console.log(textData);
    $.ajax({
        url:"http://54.183.28.228:5000/resoomer",
        data: {
          pdfData: textData
        },
        method: "POST",
        // dataType: "json"
      }).done(function(t){
        console.log("succeeded");
        console.log(t);
        $("#summarizingResult").empty().append(t);
        $('.hover_bkgr_fricc').show();
      })
})
// function setTotalNum(num){
//     $('#cuPage').empty().append(num);
// }


  