const { dialog } = require('electron').remote;
const path = require('path');
const fs = require('fs');
var mkdirp = require('mkdirp'),
  request = require('request'),
  FormData = require('form-data'),
  async = require('async');

var textData = null;
var bookmarkArray = [];

const viewerEle = document.getElementById('viewer');
viewerEle.innerHTML = ''; // destroy the old instance of PDF.js (if it exists)
const iframe = document.createElement('iframe');
iframe.src = path.resolve(__dirname, `./pdfjsOriginal/web/viewer.html?file=${require('electron').remote.getGlobal('sharedObject').someProperty}`);

viewerEle.appendChild(iframe);

filepath = require('electron').remote.getGlobal('sharedObject').someProperty;

deepai.setApiKey('a5c8170e-046a-4c56-acb1-27c37049b193');
//get text from pdf to send to flask backends
var PDF_URL  = filepath;
console.log(PDF_URL);
var capeClicked = false;
var btnClicked = false;
var bookmarkOpened = false;

///HTMLLLINNNGGGG
pdfAllToHTML(PDF_URL);
// let thedir = "";
// thedir += nameOfFileDir;
// let nameofFile = thedir.substring(thedir.lastIndexOf("\\") + 1, thedir.lastIndexOf("."));
// console.log(thedir);
// thedir = thedir.substring(0, thedir.lastIndexOf("\\") + 1);
// console.log(thedir);
//
// let inputfile = thedir + nameofFile + ".pdf";
// console.log(inputfile);
// let outputfile = thedir + nameofFile + ".html";
// let imagedir = thedir;

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
          console.log(pdfPage);
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

//summarization function
$('#summarizingButton').click(function(){
  console.log("summarizingButtonClicked");
  console.log($('#topageRange').val());
  $('.su_popup').hide();
  summaryButtonPressed($('#pageRange').val(),$('#topageRange').val());
  ////////////////////////////getSeletedPageTextForSummarization($('#pageRange').val(),$('#topageRange').val());
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

// function getTextByPageForSummarization(instance){
//   getPageText(iPagesum , instance).then(function(textPage){
//     if(iPagesum != 0)
//       textDsum += textPage
//     if(iPagesum < iEndPagesum){
//       iPagesum++;
//       getTextByPageForSummarization(instance)
//     }else{
//       deepai.setApiKey('a5c8170e-046a-4c56-acb1-27c37049b193');
//       console.log(textDsum);
//
//       var start;
//       var options = init();
//       start = new Date()
//       //processGROBID(options);
//
//
//       return;
//     }
//   });
// }

//
// /* Module Require */
//
// // the URL of the GROBID service (to be changed if necessary)
// //const GROBID_URL = "http://localhost:8070/api/";
//
// // for making console output less boring
// const green = '\x1b[32m';
// const red = '\x1b[31m';
// const orange = '\x1b[33m';
// const white = '\x1b[37m';
// const blue = `\x1b[34m`;
// const score = '\x1b[7m';
// const bright = "\x1b[1m";
// const reset = '\x1b[0m';
//
// /**
//  * List all the PDF files in a directory in a synchronous fashion,
//  * @return the list of file names
//  */
//
// function callGROBID(options, file, callback) {
//     console.log("---\nProcessing: " + PDF_URL);
//
//     var form = new FormData();
//     form.append("input", fs.createReadStream(PDF_URL));
//     form.append("consolidateHeader", "1");
//     form.append("consolidateCitations", "0");
//     console.log(typeof fs.createReadStream(options.inPath+"/"+file));
//     var grobid_url = "http://" + options.grobid_host;
//     if (options.grobid_port)
//         grobid_url += ':' + options.grobid_port
//     grobid_url += '/api/';
//     form.submit(grobid_url+options.action, function(err, res, body) {
//         if (err) {
//             console.log(err);
//             return false;
//         }
//
//         if (!res) {
//             console.log("GROBID service appears unavailable");
//             //return false;
//         } else {
//            res.setEncoding('utf8');
//         }
//
//         if (res.statusCode == 503) {
//             // service unavailable, normally it means all the threads for GROBID on the server are currently used
//             // so we sleep a bit before retrying the process
//             sleep.sleep(options.sleep_time);
//             return callGROBID(options, file, callback);
//         } else if (res.statusCode == 204) {
//             // success but no content, no need to read further the response and write an empty file
//             return true;
//         } else if (res.statusCode != 200) {
//             console.log("Call to GROBID service failed with error " + res.statusCode);
//             return false;
//         }
//
//         var body = "";
//         res.on("data", function (chunk) {
//             body += chunk;
//         });
//
//         res.on("end", function () {
//             mkdirp(options.outPath, function(err, made) {
//                 // I/O error
//                 if (err)
//                     return cb(err);
//                 console.log(body);
//                 // first write the TEI reponse
//
//                 let lessthan = body.indexOf('<p');
//                 while (lessthan !== -1) {
//                   body = body.slice(0,lessthan) + "!br!" + body.slice(body.indexOf('>',lessthan) + 1);
//                   lessthan = body.indexOf('<p');
//                 }
//                 lessthan = body.indexOf('<');
//                 while (lessthan !== -1) {
//                   body = body.slice(0,lessthan) + " " + body.slice(body.indexOf('>',lessthan) + 1);
//                   lessthan = body.indexOf('<');
//                 }
//                 String.prototype.replaceAll = function(search, replacement) {
//                   var target = this;
//                   return target.replace(new RegExp(search, 'g'), replacement);
//                 }
//                 body = body.replaceAll("!br!","<br>");
//                 deepai.callStandardApi("summarization", {
//                   text: body}).then((resp) => processSummarizationResult(resp));
//             });
//         });
//     });
// }
//
// /**
//  * Process a PDF file by calling the entity-fishing service and enrich with the resulting
//  * JSON
//  * @param {object} options object containing all the information necessary to manage the paths:
//  *  - {object} inPath input directory where to find the PDF files
//  *  - {object} outPath output directory where to write the results
//  *  - {string} profile the profile indicating which filter to use with the entity-fishing service, e.g. "species"
//  * @return {undefined} Return undefined
//  */
// function processGROBID(options) {
//     // get the PDF paths
//
//     var q = async.queue(function (file, callback) {
//         callGROBID(options, file, callback);
//     }, options.concurrency);
//
//     q.drain = function() {
//         console.log(red, "\nall tasks completed!", reset);
//         end();
//     }
//     q.push("/LaTeX_template_for_preparing_supplementary_material_for_submission_to_Optica.pdf");
// }

function pdfAllToHTML(nameOfFileDir) {
  var exec = require('child_process').exec, child;
  //update the directory to the correct one for PDF and HTML Files
  let thedir = "";
  thedir += nameOfFileDir;
  let nameofFile = thedir.substring(thedir.lastIndexOf("\\") + 1, thedir.lastIndexOf("."));
  console.log(thedir);
  thedir = thedir.substring(0, thedir.lastIndexOf("\\") + 1);
  console.log(thedir);

  let inputfile = thedir + nameofFile + ".pdf";
  console.log(inputfile);
  let outputfile = thedir + nameofFile + ".html";
  let imagedir = thedir;

  //update directory to JAR file
  let executionstring = 'java -jar C:/Users/alimi/Downloads/EtudeXML/PDFToHTML.jar ' + inputfile + ' ' + outputfile;
  //+ ' -idir=' + imagedir
  console.log(executionstring);
  child = exec(executionstring,
      function (error, stdout, stderr) {
          console.log('stdout: ' + stdout);
          console.log('stderr: ' + stderr);
          if (error !== null) {
               console.log('exec error: ' + error);
          }
      });
}

function summaryButtonPressed(firstpage, lastpage) {
  htmlWholeFileToPartialPlainText(firstpage, lastpage);
}

//TO BE ADDED AFTER JAR FILE CALL TO TAKE PLAIN TEXT OUT OF HTML FILE WHEN SUMMARY OR QUESTION
function htmlWholeFileToPartialPlainText(firstpage, lastpage) {

  //repeatedcode but needed
  let thedir = "";
  thedir += PDF_URL;
  let nameofFile = thedir.substring(thedir.lastIndexOf("\\") + 1, thedir.lastIndexOf("."));
  console.log(thedir);
  thedir = thedir.substring(0, thedir.lastIndexOf("\\") + 1);
  console.log(thedir);

  let inputfile = thedir + nameofFile + ".pdf";
  console.log(inputfile);
  let outputfile = thedir + nameofFile + ".html";
  let imagedir = thedir;




  const htmlToJson = require('html-to-json');
  let bigarray = [];
  let bigarrayback = [];
  //the correct html file directory within our project
  fs.readFile(outputfile, "utf8", function(err, data) {

    let datadata = data.split("<div class=\"page\"");
    let newstring = "";
    console.log(datadata.length);
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
    console.log(bigarray[maxindex]);
    //make sure to make it utf8 afterwards!!
    // //where to put the
    // fs.writeFile('C:/Users/alimi/Downloads/EtudeXML/compartoo.txt',bigarray[maxindex], 'utf8', function(err) {
    //   console.log(err);
    // })
    deepai.callStandardApi("summarization", {
                       text: bigarray[maxindex]}).then((resp) => processSummarizationResult(resp));
  });

}


/**
 * Init the main object with paths passed with the command line
 */
// function init() {
//     var options = new Object();
//
//     // start with the config file
//     const config = require('./config.json');
//     // options.grobid_host = config.grobid_host;
//     // options.grobid_port = config.grobid_port;
//     // options.sleep_time = config.sleep_time;
//
//     // default service is full text processing
//     // options.action = "processFulltextDocument";
//     // options.concurrency = 10; // number of concurrent call to GROBID, default is 10
//     var attribute; // name of the passed parameter
//     // options.inPath = "C:/Users/alimi/Downloads"
//     // options.outPath = "C:/Users/alimi/Downloads"
//     //
//     // console.log("\nGROBID service: ", red, options.action+"\n", reset);
//     //
//     // if (!options.inPath) {
//     //     console.log("Input path is not defines");
//     //     return;
//     // }
//
//     // check the input path
//     fs.lstat(options.inPath, (err, stats) => {
//         if (err)
//             console.log(err);
//         if (stats.isFile())
//             console.log("Input path must be a directory, not a file");
//         if (!stats.isDirectory())
//             console.log("Input path is not a valid directory");
//     });
//
//     // check the output path
//     if (options.outPath) {
//         fs.lstat(options.outPath, (err, stats) => {
//             if (err)
//                 console.log(err);
//             if (stats.isFile())
//                 console.log("Output path must be a directory, not a file");
//             if (!stats.isDirectory())
//                 console.log("Output path is not a valid directory");
//         });
//     }
//     return options;
// }

function end() {
    console.info('Execution time')
}

$('#getRangeButton').click(function(){
  //close the bookmark page
  $("#bookmark_item").attr("data","true");
  $("#bookmark_item").click();

  $('#getRangeButton').hide();
  $('.su_popup').show();
})
