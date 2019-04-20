const { dialog } = require('electron').remote;
const path = require('path');
console.log("Hello world");
const deepai = require("deepai");
console.log("got deepai");
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

function proccessSummarizationResult(t){
  console.log("succeeded");
        console.log(t);
        $("#summarizingResult").empty().append(t);
        //here you can remove the loading button
        $('.summarizer_loading').hide();
        $('.hover_bkgr_fricc').show();
        iPagesum = 0;
        iEndPagesum = 0;
        textDsum = 0;
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
      //post text.
      console.log(textD);
      $.ajax({
        url:"http://54.183.6.45:5000/cape",
        data: {
          pdfData: textD,
          question: $("#questionVal").val()
        },
        method: "POST",
        // dataType: "json"
      }).done(function(t){
        console.log("succeeded");
        console.log(t);
        $("#capeResult").empty().append(t);
        // $('.hover_bkgr_fricc').show();
        document.getElementById("myDropdown").classList.toggle("show");
         //init for search
        iPage = 0;
        iEndPage = 0;
        textD = 0;
        capeClicked = true;
      })
     
      return;
    }
  });
}

function proccessSummarizationResult(t){
const { dialog } = require('electron').remote;
const path = require('path');
const deepai = requrie("deepai")
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
function processSummarizationResult(t){
  $("#summarizingResult").empty().append(t);
  //here you can remove the loading button
  $('.summarizer_loading').hide();
  $('.hover_bkgr_fricc').show();
  iPagesum = 0;
  iEndPagesum = 0;
  textDsum = 0;
}
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
      //post text.
      console.log(textD);
      $.ajax({
        url:"http://54.183.6.45:5000/cape",
        data: {
          pdfData: textD,
          question: $("#questionVal").val()
        },
        method: "POST",
        // dataType: "json"
      }).done(function(t){
        console.log("succeeded");
        console.log(t);
        $("#capeResult").empty().append(t);
        // $('.hover_bkgr_fricc').show();
        document.getElementById("myDropdown").classList.toggle("show");
         //init for search
        iPage = 0;
        iEndPage = 0;
        textD = 0;
        capeClicked = true;
      })
     
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
      deepai.setApiKey('a5c8170e-046a-4c56-acb1-27c37049b193');
      deepai.callStandardApi("summarization", {
        text: data}).then((resp) => processSummarizationResult(resp));
      // $.ajax({
      //   url:"http://54.183.6.45:5000/resoomer",
      //   data: {
      //     pdfData: textDsum
      //   },
      //   method: "POST",
      //   // dataType: "json"
      // }).done(function(t){
      //   console.log("succeeded");
      //   console.log(t);
      //   $("#summarizingResult").empty().append(t);
      //   //here you can remove the loading button
      //   $('.summarizer_loading').hide();
      //   $('.hover_bkgr_fricc').show();
      //   iPagesum = 0;
      //   iEndPagesum = 0;
      //   textDsum = 0;
      // })
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
      console.log("posting teg");
      console.log(textDsum);
      console.log(typeof(textDsum))
      deepai.setApiKey('166c6927-b9cc-463c-9b7b-e332809895c9');
      data = "The Centre for Applied Philosophy and Public Ethics Working Paper Series The Centre for   Applied Philosophy and Public Ethics (CAPPE) was established in 2000 as a Special Research   Centre in applied philosophy funded by the Australian Research Council. It has combined the   complementary strengths of two existing centres specialising in applied philosophy, namely the   Centre for Philosophy and Public Issues (CPPI) at the University of Melbourne and the Centre   for Professional and Applied Ethics at Charles Sturt University. It operates as a unified centre   with two divisions: in Melbourne at the University of Melbourne and in Canberra at Charles Sturt   University. The Director of CAPPE and the head of the Canberra node is Professor Seumas   Miller. Professor C.A.J. (Tony) Coady is the Deputy Director of CAPPE and the head of the   Melbourne node.     The Centre concentrates, in a single unit, the expertise of applied philosophers working in   diverse fields of research. The Centre promotes community discussion and professional   dialogue concerning key ethical problems facing Australia today. It is Australia's leading centre   of research excellence in this area and it has extensive links with international institutions and   scholars. The Centre has also established collaborative projects with a number of Australian   and overseas universities. The Melbourne division of the Centre, in its previous form as CPPI,   has conducted business ethics consultancies and roundtables with many of Australia's leading   companies, including Shell, Telstra, BHP, Sydney Water and Western Mining. Such activities   continue in CAPPE - it sponsors workshops, conferences and public lectures on topics of   current public interest. The Occasional Paper Series was developed by CAPPE as a forum for   the dissemination of ideas prior to publication in academic journals or books. It is hoped that   publication in this series will elicit constructive criticisms and comments from specialists in the   relevant field. Inclusion in this series does not preclude publication elsewhere.     Michael O’Keefe, Series Editor © Robert Sparrow, 2002. Published by the Centre for Applied   Philosophy and Public Ethics, 2002.     Draft Only: Not to be cited without permission. The March of the Robot Dogs Abstract: Following   the success of Sony Corporation’s “AIBO”, robot cats and dogs are multiplying rapidly. “Robot   pets” employing sophisticated artificial intelligence and animatronic technologies are now being   marketed as toys and companions by a number of large consumer electronics corporations.     It is often suggested in popular writing about these devices that they could play a worthwhile   role in serving the needs of an increasingly aging and socially isolated population. Robot   companions, shaped like familiar household pets, could comfort and entertain lonely older   persons. This goal is misguided and unethical. While there are a number of apparent benefits   that might be thought to accrue from ownership of a robot pet, the majority and the most   important of these are predicated on mistaking, at a conscious or unconscious level, the robot   for a real animal. For an individual to benefit significantly from ownership of a robot pet they   must systematically delude themselves regarding the real nature of their relation with the   animal. It requires sentimentality of a morally deplorable sort. Indulging in such sentimentality   violates a (weak) duty that we have to ourselves to apprehend the world accurately. The design   and manufacture of these robots is unethical in so far as it presupposes or encourages this   delusion. The invention of robot pets heralds the arrival of what might be called “ersatz   companions” more generally. That is, of devices that are designed to engage in and replicate   significant social and emotional relationships. The advent of robot dogs offers a valuable   opportunity to think about the worth of such companions, the proper place of robots in society   and the value we should place on our relationships with them. Introduction1 For decades now,   pundits have been predicting the presence of robots in the homes of the future. The invention of   tireless household robots was supposed to free us from the demands of domestic drudgery and   lead us into a brave new world of greatly increased leisure time. This future has been resolutely   slow to arrive. The technical demands of performing useful tasks in a chaotic environment of   uneven surfaces alongside human beings has proved more difficult than robot enthusiasts   accounted for. The market for household robots has also been severely constrained by the fact   that, for the foreseeable future at least, it seems likely to remain far cheaper to employ cheap   human labour to do the housework than to purchase an expensive robot. Until recently, robots   have been confined to industrial or, occasionally, military or exploratory applications. However,   in the last two years or so, robots have finally begun to appear in homes - in the somewhat   surprising shape of robot pets! Following the success of Sony’s “AIBO”, robot cats and dogs are   multiplying rapidly. “Entertainment robotics” is widely anticipated as a burgeoning field.2 At first   sight the idea of robot pets seems relatively innocuous. They are but one of a range of diverting   new technological entertainments made possible by improvements in computing technology.   But in the search for a more noble purpose for their research – and, more cynically, in search for   more funding – a number of researchers have seized on the idea that such devices could play a   worthwhile role in serving the needs of an increasingly aging and socially isolated population.3   Robot companions, shaped like familiar household pets, could comfort and entertain lonely   older persons. 1 I would like to thank Jeremy Aarons, Andrew Alexandra, Jacqui Broad and   Kate Crawford for discussion and comments over the course of the development of this paper. 2   It is clear that some robot manufacturers hope that the development of robot pets will greatly   accelerate the acceptance of robots into the home. Playing with them will accustom us to   robots, while the technology developed for them can also be applied in household robots with   more ambitious purposes. See Fujita, M. and H. Kitano. Development of an Autonomous   Quadruped Robot for Robot Entertainment. Autonomous Robots, 5:7-18, 1998. 3 See, for   instance, “Glimpses of a robotic future”,   http://news.bbc.co.uk/hi/english/world/asiapacific/newsid_1048000/1048602.stm, at 15.02.02;   “Robot Dog a Japanese Techno-sensation”,   http://augustachronicle.com/stories/051699/tec_robot.shtml at 14.02.02; Irene M. Kunii. How   much is that Robot in the Window? Business Week: Asian Edition, November 27, 2000, page   22; Yuri Kageyama. Nurse Gadget patrols the wards. The Age, April 6, 2002, page 44,   Melbourne, Australia. For a useful survey of the cutting edge of contemporary robotics research,   which highlights Japanese interest, especially, in robots as carers and companions for the   elderly, see Menzel, P. and F. D'Aluisio. Robo Sapiens: Evolution of a New Species. The MIT   Press, Cambridge, Mass., 2000. It is clear that this In this paper I argue that this goal is   misguided and unethical. While there are a number of apparent benefits that might be thought to   accrue to the lonely aged from the ownership of a robot pet, the majority and the most important   of these are predicated on them mistaking, at a conscious or unconscious level, the robot for a   real animal. For an individual to benefit significantly from ownership of a robot pet they must   systematically delude themselves regarding the real nature of their relation with the animal. It   requires sentimentality of a morally deplorable sort. Indulging in such sentimentality violates a   (weak) duty that we have to ourselves to apprehend the world accurately. The design and   manufacture of these robots is unethical in so far as it presupposes or encourages this delusion.     The evil of robot pets is not the most urgent issue facing society today. It is far from being the   most significant ethical issue arising out of our treatment of the growing numbers of older   persons in our community. It may therefore seem an odd topic for philosophical treatment. But   the invention of robot pets, and the suggestion that they could play a worthwhile role as   companions for the lonely aged, heralds the arrival of what might be called “ersatz companions”   more generally. That is, of devices that are designed to engage in and replicate social and   emotional relationships of sorts that we value. In the future, the attempt will perhaps be made to   develop robot companions in the shape of human beings – “androids”.4 The advent of robot   dogs offers a valuable opportunity to think about the worth of such companions, the place that   robots might take in human society and the value we should place on our relationships with   them. I hope that the conclusions of the paper will therefore be relevant to a much wider range   of issues that are likely to arise as these technologies insinuate themselves further into our   society.     The March of the Robot Dogs   To those not familiar with the rhetoric of the roboticists and their marketing gurus it may seem   farcical that anyone should take the idea of developing robot companions for the elderly   seriously. But not only have a number of robot designers and developers publicly justification for   robot research genuinely represents an influential vision of the future application of robots,   which is likely to come true to some extent over the coming years. 4 A rather naive discussion of   the possibility of android companions may be found in Chp 6, ‘Surrogate People” of Geoff   Simons. Robots: The Quest for Living Machines. Cassell, London, 1992, pages 166-193. This   discussion includes an enthusiastic endorsement of the possibility of robot lovers! expressed   their interest and involvement in this project but there are already a number of robot pets on the   market - and more are on their way.     The most widely known and probably most advanced robot pet is a robot dog marketed by Sony   Corporation called AIBO.5 AIBO is an acronym for Artificial Intelligence roBOt but also, tellingly,   a Japanese word that is variously translated as “friend”, “partner” or “buddy”. AIBO is a   sophisticated “entertainment robot” that makes use of near state-ofthe-art artificial intelligence   and robotics technology in the attempt to generate complex behaviour in a robot that will   (hopefully) entertain and amuse those around it. AIBO has a sense of touch, hearing, sight and   a sense of balance. He can walk, shake hands, chase a ball and even dance.6 AIBO has   programmed instincts, or drives including: Movement, Fear, Recharge, and Search. AIBO can   also express six “emotions”: happiness, anger, fear, sadness, surprise and dislike. He   expresses his emotional state with a wag of his tail or by changing the colour and shape of his   eyes or by his body movements. He also barks, whines, growls and uses a series of musical   tones to fully express his mood. The latest version of AIBO, AIBO ERS-210, has voice   recognition and can understand up to 50 voice commands. Once you have recorded a name for   your robot companion, AIBO is able to recognise it, and will respond with electronic tones when   he is called. You can tell AIBO to dance, sit, or to take a picture of you with the digital camera   located in his nose.     The combination of AIBO’s drives, emotions and stimulus produces “behaviour”, which is   accordingly relatively complex and unpredictable. AIBO’s behaviour is also dependent on his   interaction with his owner and therefore, according to Sony’s promotional material, no two   AIBOs are ever alike. He grows and develops as time passes and according to how much he is   played with, proceeding through the developmental stages of an infant, child, teen and adult.   The type and amount of attention his owner gives AIBO, will determine his personality which in   turn influences behaviour. AIBO even has the ability to “learn” and “unlearn” certain behaviour.7   5 Sony has actually released three versions of AIBO, each more sophisticated than the last.   According to one source, Panasonic are apparently also developing robot teddy bears and cars   designed as companions for old people. See “Robot Dog a Japanese Techno-sensation”,   http://augustachronicle.com/stories/051699/tec_robot.shtml at 14.02.02. 6 Marketing for, and   media reportage of, AIBO typically genders “him” as male. 7 My description of AIBO’s   capabilities is taken more or less verbatim from various promotional materials published by   Sony on the Web. See, for instance, “AIBO Homepage”, http://www.us.aibo.com, at 14.02.02;   “Sony AIBO Robot Dog”, http://www.robotbooks.com/sony_aibo.htm at 16.8.01; “AIBO   Homepage”, http://www.eu.aibo.com at 16.8.01. There is undoubtedly something toylike or   gadget-like about AIBO. AIBO’s moulded plastic surfaces and mechanical gait leave little room   for the illusion that it is alive.8 He looks like a robot dog rather than a dog and his design   appeals to cultural archetypes of robots perpetuated through representation in cartoons,   television and film.9 There are also various accessories that one can purchase to extend his   capabilities and increase his range of behaviours, including memory cards that allow him to   mature and develop, or alternatively become an “adult” dog instantaneously, and one that   allows him to play “scissors, paper, rock”. There is even a special carry bag available to   transport him. The existence and marketing of these accessories makes it even more obvious   that we are dealing with a clever gadget rather than a real animal.10 One suspects that the   majority of the people who have purchased AIBO (some 90 000 to date) do so in the belief that   they are buying a cool toy rather than acquiring a robot companion.     A “friend for life”   Despite this, Sony’s promotional material is adamant that AIBO is not a toy and states so   explicitly and repeatedly. According to one corporate source, “AIBO is not a toy! He is a true   companion with real emotions and instincts. With loving attention from his master, he will   develop into a more mature and fun- loving friend as time passes.”11 And also, “Like any   human or animal, AIBO goes through the developmental stages of an infant, child, teen and   adult. Daily communication and attention will determine how he matures. The more interaction   you have with him, the faster he grows up. In short, AIBO is a friend for life.”12 8 Although as   we shall see below, the relative complexity of his behaviour is likely to cause at least some   people to attribute emotional states to it. 9 In fact while the first two versions of AIBO were   modelled on dogs, the latest version is apparently modelled on a lion club. This latter design   apparently allows those who wish to identify AIBO with a cat the latitude to do so. 10 A   flourishing subculture has even grown up around modifying AIBO and altering its programming.   Such AIBO “hackers” presumably have no illusions that they are dealing with a creature with   “real emotions”, instead they are experimenting with a new technology and seeing what   possibilities it offers. 11 “Sony AIBO Robot Dog”, http://www.robotbooks.com/sony_aibo.htm, at   16.8.01. 12 “Sony AIBO Robot Dog”, http://www.robotbooks.com/sony_aibo.htm, at 16.8.01.   AIBO is intended and advertised as a “robot companion”. Indeed Sony Australia’s AIBO website   is titled “AIBO – Your companion for the new millennium”. No doubt much of this is marketing   hype. One doubts that AIBO’s design team think of him as a “friend for life”. Yet Sony obviously   believes that it can succeed in promoting AIBO, to some people at least, as a companion and a   substitute for a real pet.     Other robot pets   As well as AIBO there are at least 13 other sorts of “robot pets” currently on sale around the   world, including “Poo-Chi” and “Meow-chi” (a robot cat), “Tekno the Robot Puppy” and “Kitty the   Tekno Kitten”, “Tiny the Tekno Puppy”, “Super Poo-Chi”, “Furby” (a robot cat), “I-Cybie” (a robot   dog), “NeCoRo” (a robot cat) “Big and Lil’ Scratch” (dogs again), “Rocket the Wonder Dog” and   “Baby Rocket Puppy”.13 Most of these are much more obviously toys than “robot companions”   (and are consequently much cheaper than AIBO). Nonetheless they also are designed to   “interact” with their owners to some extent and have primitive personalities, sets of behaviours   and learning mechanisms.14 Their marketing emphasises their interactive nature, their ability to   learn and their ability to demonstrate and express emotions. In several cases it is suggested   that these pets can become your “friend”.15 Other robot pets are under development. Some of   these will undoubtedly outdo AIBO in terms of complexity and range of behaviours. “My Real   Baby” It is also worth mentioning at this point a related product, although again more clearly   intended as a toy than as a substitute companion; American toy company Hasbro’s, “My Real   Baby”. Produced in collaboration with the robot manufacturer iRobot, “My Real Baby” is a life   sized baby doll which makes use of artificial intelligence technology and advanced   “animatronics” in order to generate a wide range of facial expressions and behaviour. Like   AIBO, My Real Baby responds to and learns from its owner’s treatment 13 Descriptions of these   robot pets (plus a few more besides!) can be found at “Robot Dogs”,   http://www.robotbooks.com/robot-dogs.htm, at 16.8.01; Michael Idato, “Living dolls”,   http://it.mycareer.com.au/techlife/inventingthefuture/2001/11/24/FFXT6464HUC.html, at   14.2.02. 14 In particular “NeCoRo” is a robot cat that is designed to establish an emotional bond   with its owner. It has much more limited abilities to move than other robot pets, but a much   greater ability to interact with its owner, through being petted and purring or stretching etc, in   order to establish a rewarding relationship. See “Robo-cat is out of the bag”,   http://news.bbc.co.uk/hi/english/sci/tech/newsid_1602000/602677.stm, at 17.10.00. 15 “Robot   Dogs”, http://www.robotbooks.com/robot-dogs.htm, at 16.8.01. of it. It possesses 15 different   “emotional states”. It can sense how it is being treated by its owner and alters its behaviours   "
      deepai.callStandardApi("summarization", {
        text: data}).then((resp) => console.log(resp));

      // .fail(function(err) {
      //   console.log("you failed so try again I know you are great so you can be best all the time, fighting!");
      // });
     
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




  