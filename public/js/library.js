const { dialog } = require('electron').remote;
const path = require('path');
const Store = require('electron-store');
var store = new Store();
const { ipcRenderer } = require('electron');
store.clear();
var i = store.size;

var __PDF_DOC,
        __CURRENT_PAGE,
        __TOTAL_PAGES,
        __PAGE_RENDERING_IN_PROGRESS = 0,
        index = 0;

        for (var j = 0; j < i; j++){
                // var j = 0;
          show_nextItem(store.get(j.toString()), j.toString());
          console.log(store.get(j.toString()));
          showPDF_fresh(store.get(j.toString()),j);

        }
        document.getElementById('myButton').addEventListener('click', () => {
                 dialog.showOpenDialog({
                    properties: ['openFile'], // set to use openFileDialog
                    filters: [ { name: "PDFs", extensions: ['pdf'] } ] // limit the picker to just pdfs
                  }, (filepaths) => {
                        var filePath = filepaths[0];
                        console.log(filePath);
                        // var sssss = filePath.replace(/\s/g, "");
                        // filePath = sssss;
                        // console.log(sssss);
                        // console.log("I printed sssssss above");
                        console.log(filePath);
                        show_nextItem(filePath,i.toString());
                        showPDF(filePath);  
                  })

        })

        function showPDF(pdf_url) {
	        PDFJS.getDocument({ url: pdf_url }).then(function(pdf_doc) {
		        __PDF_DOC = pdf_doc;
		        __TOTAL_PAGES = __PDF_DOC.numPages;
                        // Show the first page
                        showPage(1);
                        store.set(i.toString(),pdf_url);
                        i++; 
	        }).catch(function(error) {
                        alert(error.message);
	        });;
        }
        function showPDF_fresh(pdf_url, num) {
                PDFJS.getDocument({ url: pdf_url }).then(function(pdf_doc) {
                        __PDF_DOC = pdf_doc;
                        __TOTAL_PAGES = __PDF_DOC.numPages;
                        // Show the first page
                        showPage_fresh(1,num);
                        //  store.set(i.toString(),pdf_url);
                        // i++; 
                }).catch(function(error) {
                        alert(error.message);
                });;
        }
        function showPage(page_no) {
                __PAGE_RENDERING_IN_PROGRESS = 1;
                __CURRENT_PAGE = page_no;
                
                // Fetch the page
	        __PDF_DOC.getPage(page_no).then(function(page) {
                        var __CANVAS = $('.pdf-canvas').get($(".pdf-canvas").length-1),
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

                                // Re-enable Prev & Next buttons
                                // $("#pdf-next, #pdf-prev").removeAttr('disabled');

                                // Show the canvas and hide the page loader
                                $(".pdf-canvas").show();
                                // $("#page-loader").hide();
                                //$(".book_section").children("img").last().remove();
                        
                                // $('.plusImage').hide();
                                // $('.plusImage').removeClass("plusImage");
                                // show_nextItem();
                                       
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

                                // Re-enable Prev & Next buttons
                                // $("#pdf-next, #pdf-prev").removeAttr('disabled');

                                // Show the canvas and hide the page loader
                                $(".pdf-canvas").show();
                                // $("#page-loader").hide();
                                //$(".book_section").children("img").last().remove();
                        
                                // $('.plusImage').hide();
                                // $('.plusImage').removeClass("plusImage");
                                // show_nextItem();
                                       
                        });
                        // index++;
                });
        }

        function show_nextItem(pdf_path,removeWhich)
        {
                // var name = pdf_path.split('/');
                console.log(pdf_path);
                console.log("inside insde the show_nextItem");
                // we get the name of the pdf
                var filenamewithextension = path.parse(pdf_path).base;
                var filename = filenamewithextension.split('.')[0];
                console.log(filename);
                var next_text = "<div class='col-md-2 book_section'><div><center><canvas class='pdf-canvas' data ='"+ pdf_path + "' id = 'viewer'></canvas><img class = 'minusImage' data ="+ removeWhich + " id = 'myButton' src='./public/images/cross.png'/><p style = 'width: 250px; word-break: break-all;'>"+filename+"</p></center></div></div>";
                var next_div = document.createElement("div");
                next_div.innerHTML = next_text;
                document.getElementById('container').append(next_div);
        }

        //when the user select the pdf
        $(document).on("click",".pdf-canvas", function(){
          console.log($(this).attr("data"));
          console.log("above you clicked sth");
          ipcRenderer.send('show_pdf_message', $(this).attr("data"));
          window.location.href = 'summarizing.html';
        });
        // when the user click the minus button
        $(document).on("click",".minusImage", function(){
                ($(this).parent()).parent().remove();
                //delete it in the store
                store.delete($(this).attr("data"));
                // sort the store
                  for (var k = parseInt($(this).attr("data")); k < store.size; k++)
                  {
                    store.set(k.toString(),store.get((k+1).toString()));
                    store.delete((k+1).toString());
                  }

        });