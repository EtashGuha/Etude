const { dialog } = require('electron').remote;
const path = require('path');

// java.classpath.push("Kernel.jar");
// java.classpath.push("/Applications/Wolfram\ Desktop.app/Contents/SystemFiles/Links/JLink/JLink.jar")
// var kernel = java.newInstanceSync('p1.Kernel');
// const Storebookmark = require('electron-store');
// var bookmarkStore = new Storebookmark();
// bookmarkStore.clear();
// var i = bookmarkStore.size;
var textData = null;
var bookmarkArray = [];

const viewerEle = document.getElementById('viewer');
viewerEle.innerHTML = ''; // destroy the old instance of PDF.js (if it exists)
const iframe = document.createElement('iframe');
iframe.src = path.resolve(__dirname, `./pdfjsOriginal/web/viewer.html?file=${require('electron').remote.getGlobal('sharedObject').someProperty}`);

  // console.log(document.getElementsByTagName('iframe')[0].contentWindow.document.getElementById('pageNumber').value);

  // document.getElementsByTagName('iframe')[0].contentWindow.document.getElementById('thumbnailView').children[pange_number].click()


// Add the iframe to our UI.
// console.log($('#pageNumber').val());
// console.log("pagenumber is printed ddddddd");
viewerEle.appendChild(iframe);

filepath = require('electron').remote.getGlobal('sharedObject').someProperty;
   //get text from pdf to send to flask backends
var PDF_URL  = filepath;
var capeClicked = false;
var btnClicked = false;
var bookmarkOpened = false;
var java = require('java');
java.classpath.push("./Kernel.jar");
java.classpath.push("/Applications/Wolfram\ Desktop.app/Contents/SystemFiles/Links/JLink/JLink.jar")
var kernel = java.newInstanceSync('p1.Kernel');

// for (var j = 0; j < i; j++){
//         // var j = 0;
//   show_nextItem(bookmarkStore.get(j.toString()), j.toString());
//   console.log(bookmarkStore.get(j.toString()));
//   showPDF(filepath,bookmarkStore.get(j.toString()));

// }

$("#bookmark_icon").click(function(){

  // document.getElementById("bookmark_icon").src="./assets/images/bookmarkselected.png";
  // var whichpagetobookmark = $("#bookmark_select_page").val();
  //get the page number
  var whichpagetobookmark = document.getElementsByTagName('iframe')[0].contentWindow.document.getElementById('pageNumber').value;
  console.log(bookmarkArray.length);
  for (var j = 0; j < bookmarkArray.length; j++){
    if (bookmarkArray[j] == whichpagetobookmark)
      return;
  }
  show_nextItem(whichpagetobookmark,null);
  //store the page number in the database
  // bookmarkStore.set(i.toString(),whichpagetobookmark);
  bookmarkArray.push(whichpagetobookmark);
  console.log("kan kan below");
  showPDF(filepath,parseInt(whichpagetobookmark));
})

function showPDF(pdf_url,bookmark_page) {
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


// function setTotalNum(num){
//     $('#cuPage').empty().append(num);
// }
//help page show
$("#help").click(function(){
  // close the bookmark page
  $("#bookmark_item").attr("data","true");
  $("#bookmark_item").click();

  $('.help_popup').show();
	$('.help_popup').click(function(){
    $('.help_popup').hide();
  });
  // $('.popupCloseButton').click(function(){
  //   $('.hover_bkgr_fricc').hide();
  // });
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
      $("#capeResult").empty().append(kernel.findTextAnswerSync(textD, $("#questionVal").val(), 2, "Sentence"));
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



// document.body.onclick = function(e){
//   document.getElementById("myDropdown").classList.toggle("show");
// };

// here is the part of bookmark

//summarization function
$('#summarizingButton').click(function(){
  console.log("summarizingButtonClicked");
  $('.su_popup').hide();
  getSeletedPageTextForSummarization($('#pageRange').val(),$('#topageRange').val());
   
  // $('.hover_bkgr_fricc').show();
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
function getTextByPageForSummarization(instance){
  getPageText(iPagesum , instance).then(function(textPage){
    if(iPagesum != 0)
      textDsum += textPage
    if(iPagesum < iEndPagesum){
      iPagesum++;
      getTextByPageForSummarization(instance)
    }else{
      //post text.
      console.log(textDsum);
      $.ajax({
        url:"http://54.183.6.45:5000/resoomer",
        data: {
          pdfData: textDsum
        },
        method: "POST",
        // dataType: "json"
      }).done(function(t){
        console.log("succeeded");
        console.log(t);
        $("#summarizingResult").empty().append(t);
        //here you can remove the loading button
        $('.summarizer_loading').hide();
        $('.hover_bkgr_fricc').show();
        iPagesum = 0;
        iEndPagesum = 0;
        textDsum = 0;
      })
      // .fail(function(err) {
      //   console.log("you failed so try again I know you are great so you can be best all the time, fighting!");
      // });
     
      return;
    }
  });
}

$('#getRangeButton').click(function(){
  // if(btnClicked == false){
  //   $('#getRangeButton').hide();
  //   $('.su_popup').show();
  //   btnClicked = true;
  // }
  // else{
  //    $('.su_popup').hide();
  //    btnClicked = false;
  // }
  // $('.su_popup').click(function(){
  //   $('.su_popup').hide();
  // });
  //close the bookmark page
  $("#bookmark_item").attr("data","true");
  $("#bookmark_item").click();

  $('#getRangeButton').hide();
  $('.su_popup').show();
})

kernel.findTextAnswerSync('foo','bar', 1, "Sentence");
console.log('hello');
// function queueRenderPage(num) {
//   if (pageRendering) {
//     pageNumPending = num;
//   } else {
//     renderPage(num);
//   }
// }

// function onNextPage() {
//   // if (pageNum >= pdfDoc.numPages) {
//   //   return;
//   // }
//   // pageNum++;
//   queueRenderPage(3);
// }
// document.getElementById('next').addEventListener('click', onNextPage);

//bookmark open and close
// $('#bookmark_item').click(function(){

//   if(bookmarkOpened == false)
//   {
//     document.getElementById("bookmark_page").style.display = "block";
//     $('.bookmark-canvas').show();
//     $('.deleteImage').show();
//     bookmarkOpened = true;
//   }
//   if (bookmarkOpened == true)
//   {
//     document.getElementById("bookmark_page").style.height = "0";
//     $('.bookmark-canvas').hide();
//     $('.deleteImage').hide();
//     bookmarkOpened = false;
//   }

// })




  