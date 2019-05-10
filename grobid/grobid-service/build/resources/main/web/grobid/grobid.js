/**
*  Javascript functions for the front end.
*
*  Author: Patrice Lopez
*/

//jQuery.fn.prettify = function () { this.html(prettyPrintOne(this.html(),'xml')); };

var grobid = (function($) {

	var teiToDownload;
	var teiPatentToDownload;

	var block = 0;

	function defineBaseURL(ext) {
		var baseUrl = null;
		if ( $(location).attr('href').indexOf("index.html") != -1) {
            baseUrl = $(location).attr('href').replace("index.html", "api/" + ext);
        } else {
			baseUrl = $(location).attr('href') + "api/" + ext;
		}
		console.log("BaseURL: " + baseUrl);
		return baseUrl;
	}

	function setBaseUrl(ext) {
		var baseUrl = defineBaseURL(ext);
		if (block == 0)
			$('#gbdForm').attr('action', baseUrl);
		else if (block == 1)
			$('#gbdForm2').attr('action', baseUrl);
		else if (block == 2)
			$('#gbdForm3').attr('action', baseUrl);
	}

	$(document).ready(function() {
		$("#subTitle").html("About");
		$("#divAbout").show();
		//$("#divAdmin").hide();

		// for TEI-based results
        $("#divRestI").hide();

        // for PDF based results
        $("#divRestII").hide();

        // for patent processing
        $("#divRestIII").hide();

		$("#divDoc").hide();
		$('#consolidateBlock').show();
        $("#btn_download").hide();
        $("#btn_download3").hide();

		createInputFile();
		createInputFile2();
		createInputFile3();
		setBaseUrl('processHeaderDocument');
		block = 0;

		$('#selectedService').change(function() {
			processChange();
			return true;
		});

		$('#selectedService2').change(function() {
			processChange();
			return true;
		});

		$('#selectedService3').change(function() {
			processChange();
			return true;
		});

		$('#gbdForm').ajaxForm({
            beforeSubmit: ShowRequest1,
            success: SubmitSuccesful,
            error: AjaxError1,
            dataType: "text"
        });

		$('#submitRequest2').bind('click', submitQuery2);
		$('#submitRequest3').bind('click', submitQuery3);

		// bind download buttons with download methods
		$('#btn_download').bind('click', download);
		$("#btn_download").hide();
		$('#btn_download3').bind('click', downloadPatent);
		$("#btn_download3").hide();
        $('#btn_block_1').bind('click', downloadVisibilty);
        $('#btn_block_3').bind('click', downloadVisibilty3);
		//$('#adminForm').attr("action", defineBaseURL("allProperties"));
		//$('#TabAdminProps').hide();
		/*$('#adminForm').ajaxForm({
	        beforeSubmit: adminShowRequest,
	        success: adminSubmitSuccesful,
	        error: adminAjaxError,
	        dataType: "text"
	        });*/

		$("#about").click(function() {
			$("#about").attr('class', 'section-active');
			$("#rest").attr('class', 'section-not-active');
			$("#pdf").attr('class', 'section-not-active');
			//$("#admin").attr('class', 'section-not-active');
			$("#doc").attr('class', 'section-not-active');
			$("#patent").attr('class', 'section-not-active');

			$("#subTitle").html("About");
			$("#subTitle").show();

			$("#divAbout").show();
			$("#divRestI").hide();
			$("#divRestII").hide();
			$("#divRestIII").hide();
			//$("#divAdmin").hide();
			$("#divDoc").hide();
			$("#divDemo").hide();
			return false;
		});
		$("#rest").click(function() {
			$("#rest").attr('class', 'section-active');
			$("#pdf").attr('class', 'section-not-active');
			$("#doc").attr('class', 'section-not-active');
			$("#about").attr('class', 'section-not-active');
			//$("#admin").attr('class', 'section-not-active');
			$("#patent").attr('class', 'section-not-active');

			$("#subTitle").hide();
			block = 0;
			//$("#subTitle").html("TEI output service");
			//$("#subTitle").show();
			processChange();

			$("#divRestI").show();
			$("#divRestII").hide();
			$("#divRestIII").hide();
			$("#divAbout").hide();
			$("#divDoc").hide();
			//$("#divAdmin").hide();
			$("#divDemo").hide();
			return false;
		});
		/*$("#admin").click(function() {
			$("#admin").attr('class', 'section-active');
			$("#doc").attr('class', 'section-not-active');
			$("#about").attr('class', 'section-not-active');
			$("#rest").attr('class', 'section-not-active');
			$("#pdf").attr('class', 'section-not-active');
			$("#patent").attr('class', 'section-not-active');

			$("#subTitle").html("Admin");
			$("#subTitle").show();
			setBaseUrl('admin');

			$("#divRestI").hide();
			$("#divRestII").hide();
			$("#divRestIII").hide();
			$("#divAbout").hide();
			$("#divDoc").hide();
			//$("#divAdmin").show();
			$("#divDemo").hide();
			return false;
		});*/
		$("#doc").click(function() {
			$("#doc").attr('class', 'section-active');
			$("#rest").attr('class', 'section-not-active');
			$("#pdf").attr('class', 'section-not-active');
			$("#patent").attr('class', 'section-not-active');
			$("#about").attr('class', 'section-not-active');
			//$("#admin").attr('class', 'section-not-active');

			$("#subTitle").html("Doc");
			$("#subTitle").show();

			$("#divDoc").show();
			$("#divAbout").hide();
			$("#divRestI").hide();
			$("#divRestII").hide();
			$("#divRestIII").hide();
			//$("#divAdmin").hide();
			$("#divDemo").hide();
			return false;
		});
		$("#pdf").click(function() {
			$("#pdf").attr('class', 'section-active');
			$("#rest").attr('class', 'section-not-active');
			$("#patent").attr('class', 'section-not-active');
			$("#about").attr('class', 'section-not-active');
			//$("#admin").attr('class', 'section-not-active');
			$("#doc").attr('class', 'section-not-active');

			block = 1;
			setBaseUrl('referenceAnnotations');
			$("#subTitle").hide();
			processChange();
			//$("#subTitle").html("PDF annotation services");
			//$("#subTitle").show();

			$("#divDoc").hide();
			$("#divAbout").hide();
			$("#divRestI").hide();
			$("#divRestII").show();
			$("#divRestIII").hide();
			//$("#divAdmin").hide();
			return false;
		});
		$("#patent").click(function() {
			$("#patent").attr('class', 'section-active');
			$("#rest").attr('class', 'section-not-active');
			$("#pdf").attr('class', 'section-not-active');
			$("#about").attr('class', 'section-not-active');
			//$("#admin").attr('class', 'section-not-active');
			$("#doc").attr('class', 'section-not-active');

			block = 2;
			setBaseUrl('processCitationPatentST36');
			$("#subTitle").hide();
			processChange();

			$("#divDoc").hide();
			$("#divAbout").hide();
			$("#divRestI").hide();
			$("#divRestII").hide();
			$("#divRestIII").show();
			//$("#divAdmin").hide();
			return false;
		});
	});

	function ShowRequest1(formData, jqForm, options) {
	    $('#requestResult').html('<font color="grey">Requesting server...</font>');
	    return true;
	}

	function ShowRequest2(formData, jqForm, options) {
	    $('#infoResult2').html('<font color="grey">Requesting server...</font>');
	    return true;
	}

	function ShowRequest3(formData, jqForm, options) {
	    $('#requestResult3').html('<font color="grey">Requesting server...</font>');
	    return true;
	}

	function AjaxError1(jqXHR, textStatus, errorThrown) {
		$('#requestResult').html("<font color='red'>Error encountered while requesting the server.<br/>"+jqXHR.responseText+"</font>");
		responseJson = null;
	}

    function AjaxError2(message) {
    	if (!message)
    		message ="";
    	message += " - The PDF document cannot be annotated. Please check the server logs.";
    	$('#infoResult2').html("<font color='red'>Error encountered while requesting the server.<br/>"+message+"</font>");
		responseJson = null;
        return true;
    }

	function AjaxError3(jqXHR, textStatus, errorThrown) {
		$('#requestResult3').html("<font color='red'>Error encountered while requesting the server.<br/>"+jqXHR.responseText+"</font>");
		responseJson = null;
	}

	function htmll(s) {
    	return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  	}

	function SubmitSuccesful(responseText, statusText, xhr) {
		//var selected = $('#selectedService option:selected').attr('value');
		var display = "<pre class='prettyprint lang-xml' id='xmlCode'>";
		var testStr = vkbeautify.xml(responseText);
        teiToDownload = responseText;
		display += htmll(testStr);

		display += "</pre>";
		$('#requestResult').html(display);
		window.prettyPrint && prettyPrint();
		$('#requestResult').show();
        $("#btn_download").show();
	}

    function submitQuery2() {
        var selected = $('#selectedService2 option:selected').attr('value');
        if (selected == 'annotatePDF') {
            // we will have a PDF back
            //PDFJS.disableWorker = true;

            var form = document.getElementById('gbdForm2');
            var formData = new FormData(form);
            var xhr = new XMLHttpRequest();
            var url = $('#gbdForm2').attr('action');
            xhr.responseType = 'arraybuffer';
            xhr.open('POST', url, true);
            ShowRequest2();
            xhr.onreadystatechange = function (e) {
                if (xhr.readyState == 4) {
                    if (xhr.status == 200) {
                        var response = e.target.response;
                        var pdfAsArray = new Uint8Array(response);
                        // Use PDFJS to render a pdfDocument from pdf array
                        var frame = '<iframe id="pdfViewer" src="resources/pdf.js/web/viewer.html?file=" style="width: 100%; height: 1000px;"></iframe>';
                        $('#requestResult2').html();
                        $('#infoResult2').html(frame);
                        var pdfjsframe = document.getElementById('pdfViewer');
                        pdfjsframe.onload = function () {
                            pdfjsframe.contentWindow.PDFViewerApplication.open(pdfAsArray);
                        };
                    } else {
                        //AjaxError2("Response " + xhr.status + ": " + xhr.responseText);
                        AjaxError2("Response " + xhr.status + ": " );
                    }
                }
            };
            xhr.send(formData);  // multipart/form-data
        } else {
            // we will have JSON annotations to be layered on the PDF

            // request for the annotation information
            var form = document.getElementById('gbdForm2');
            var formData = new FormData(form);
            var xhr = new XMLHttpRequest();
            var url = $('#gbdForm2').attr('action');
            xhr.responseType = 'json';
            xhr.open('POST', url, true);
            ShowRequest2();

            var nbPages = -1;

            // display the local PDF
            if ((document.getElementById("input2").files[0].type == 'application/pdf') ||
                (document.getElementById("input2").files[0].name.endsWith(".pdf")) ||
                (document.getElementById("input2").files[0].name.endsWith(".PDF")))
                var reader = new FileReader();
            reader.onloadend = function () {
                // to avoid cross origin issue
                //PDFJS.disableWorker = true;
                var pdfAsArray = new Uint8Array(reader.result);
                // Use PDFJS to render a pdfDocument from pdf array
                PDFJS.getDocument(pdfAsArray).then(function (pdf) {
                    // Get div#container and cache it for later use
                    var container = document.getElementById("requestResult2");
                    // enable hyperlinks within PDF files.
                    //var pdfLinkService = new PDFJS.PDFLinkService();
                    //pdfLinkService.setDocument(pdf, null);

                    $('#requestResult2').html('');
                    nbPages = pdf.numPages;

                    // Loop from 1 to total_number_of_pages in PDF document
                    for (var i = 1; i <= nbPages; i++) {

                        // Get desired page
                        pdf.getPage(i).then(function (page) {

                            var div0 = document.createElement("div");
                            div0.setAttribute("style", "text-align: center; margin-top: 1cm;");
                            var pageInfo = document.createElement("p");
                            var t = document.createTextNode("page " + (page.pageIndex + 1) + "/" + (nbPages));
                            pageInfo.appendChild(t);
                            div0.appendChild(pageInfo);
                            container.appendChild(div0);

                            var scale = 1.5;
                            var viewport = page.getViewport(scale);
                            var div = document.createElement("div");

                            // Set id attribute with page-#{pdf_page_number} format
                            div.setAttribute("id", "page-" + (page.pageIndex + 1));

                            // This will keep positions of child elements as per our needs, and add a light border
                            div.setAttribute("style", "position: relative; border-style: solid; border-width: 1px; border-color: gray;");

                            // Append div within div#container
                            container.appendChild(div);

                            // Create a new Canvas element
                            var canvas = document.createElement("canvas");

                            // Append Canvas within div#page-#{pdf_page_number}
                            div.appendChild(canvas);

                            var context = canvas.getContext('2d');
                            canvas.height = viewport.height;
                            canvas.width = viewport.width;

                            var renderContext = {
                                canvasContext: context,
                                viewport: viewport
                            };

                            // Render PDF page
                            page.render(renderContext).then(function () {
                                // Get text-fragments
                                return page.getTextContent();
                            })
                                .then(function (textContent) {
                                    // Create div which will hold text-fragments
                                    var textLayerDiv = document.createElement("div");

                                    // Set it's class to textLayer which have required CSS styles
                                    textLayerDiv.setAttribute("class", "textLayer");

                                    // Append newly created div in `div#page-#{pdf_page_number}`
                                    div.appendChild(textLayerDiv);

                                    // Create new instance of TextLayerBuilder class
                                    var textLayer = new TextLayerBuilder({
                                        textLayerDiv: textLayerDiv,
                                        pageIndex: page.pageIndex,
                                        viewport: viewport
                                    });

                                    // Set text-fragments
                                    textLayer.setTextContent(textContent);

                                    // Render text-fragments
                                    textLayer.render();
                                });
                        });
                    }
                });
            }
            reader.readAsArrayBuffer(document.getElementById("input2").files[0]);

            xhr.onreadystatechange = function (e) {
                if (xhr.readyState == 4 && xhr.status == 200) {
                    var response = e.target.response;
                    //var response = JSON.parse(xhr.responseText);
                    //console.log(response);
                    setupAnnotations(response);
                } else if (xhr.status != 200) {
                    AjaxError2("Response " + xhr.status + ": ");
                }
            };
            xhr.send(formData);
        }
    }


	function submitQuery3() {
		var selected = $('#selectedService3 option:selected').attr('value');
		if (selected == 'citationPatentPDFAnnotation') {
			// we will have a PDF back
			//PDFJS.disableWorker = true;

			var form = document.getElementById('gbdForm3');
			var formData = new FormData(form);
			var xhr = new XMLHttpRequest();
			var url = $('#gbdForm3').attr('action');
			xhr.responseType = 'arraybuffer';
			xhr.open('POST', url, true);
			ShowRequest3();
			xhr.onreadystatechange = function(e) {
				if (xhr.readyState == 4 && xhr.status == 200) {
				    var response = e.target.response;
				    var pdfAsArray = new Uint8Array(response);
					// Use PDFJS to render a pdfDocument from pdf array
					var frame = '<iframe id="pdfViewer" src="resources/pdf.js/web/viewer.html?file=" style="width: 100%; height: 1000px;"></iframe>';
					$('#requestResult3').html(frame);
					var pdfjsframe = document.getElementById('pdfViewer');
					pdfjsframe.onload = function() {
						pdfjsframe.contentWindow.PDFViewerApplication.open(pdfAsArray);
					};
				} else  if (xhr.status != 200) {
					AjaxError3(xhr);
				}
			};
			xhr.send(formData);  // multipart/form-data
		} else if (selected == 'citationPatentAnnotations') {
			// we will have JSON annotations to be layered on the PDF

			// request for the annotation information
			var form = document.getElementById('gbdForm3');
			var formData = new FormData(form);
			var xhr = new XMLHttpRequest();
			var url = $('#gbdForm3').attr('action');
			xhr.responseType = 'json';
			xhr.open('POST', url, true);
			ShowRequest3();

			var nbPages = -1;

			// display the local PDF

			if ( (document.getElementById("input3").files[0].type == 'application/pdf') ||
			   	 (document.getElementById("input3").files[0].name.endsWith(".pdf")) ||
				 (document.getElementById("input3").files[0].name.endsWith(".PDF")) ) {
                var reader = new FileReader();
                reader.onloadend = function () {
					// to avoid cross origin issue
					//PDFJS.disableWorker = true;
				    var pdfAsArray = new Uint8Array(reader.result);
					// Use PDFJS to render a pdfDocument from pdf array
				    PDFJS.getDocument(pdfAsArray).then(function (pdf) {
				        // Get div#container and cache it for later use
			            var container = document.getElementById("requestResult3");
			            // enable hyperlinks within PDF files.
			            //var pdfLinkService = new PDFJS.PDFLinkService();
			            //pdfLinkService.setDocument(pdf, null);

						$('#requestResult3').html('');
						nbPages = pdf.numPages;

			            // Loop from 1 to total_number_of_pages in PDF document
			            for (var i = 1; i <= nbPages; i++) {

			                // Get desired page
			                pdf.getPage(i).then(function(page) {

							  	var div0 = document.createElement("div");
							  	div0.setAttribute("style", "text-align: center; margin-top: 1cm;");
			                  	var pageInfo = document.createElement("p");
			                  	var t = document.createTextNode("page " + (page.pageIndex + 1) + "/" + (nbPages));
							  	pageInfo.appendChild(t);
							  	div0.appendChild(pageInfo);
			                  	container.appendChild(div0);

			                  	var scale = 1.5;
			                 	var viewport = page.getViewport(scale);
				                var div = document.createElement("div");

			                  	// Set id attribute with page-#{pdf_page_number} format
			                  	div.setAttribute("id", "page-" + (page.pageIndex + 1));

			                  	// This will keep positions of child elements as per our needs, and add a light border
			                  	div.setAttribute("style", "position: relative; border-style: solid; border-width: 1px; border-color: gray;");

			                  	// Append div within div#container
			                  	container.appendChild(div);

			                  	// Create a new Canvas element
			                  	var canvas = document.createElement("canvas");

			                  	// Append Canvas within div#page-#{pdf_page_number}
			                  	div.appendChild(canvas);

			                  	var context = canvas.getContext('2d');
			                  	canvas.height = viewport.height;
			                  	canvas.width = viewport.width;

			                  	var renderContext = {
			                    	canvasContext: context,
			                  		viewport: viewport
			                  	};

			                  	// Render PDF page
			                  	page.render(renderContext).then(function() {
			                        // Get text-fragments
			                        return page.getTextContent();
			                    })
			                    .then(function(textContent) {
			                        // Create div which will hold text-fragments
			                        var textLayerDiv = document.createElement("div");

			                        // Set it's class to textLayer which have required CSS styles
			                        textLayerDiv.setAttribute("class", "textLayer");

			                        // Append newly created div in `div#page-#{pdf_page_number}`
			                        div.appendChild(textLayerDiv);

			                        // Create new instance of TextLayerBuilder class
			                        var textLayer = new TextLayerBuilder({
			                          textLayerDiv: textLayerDiv,
			                          pageIndex: page.pageIndex,
			                          viewport: viewport
			                        });

			                        // Set text-fragments
			                        textLayer.setTextContent(textContent);

			                        // Render text-fragments
			                        textLayer.render();
			                    });
			                });
			            }
				    });
				}
				reader.readAsArrayBuffer(document.getElementById("input3").files[0]);
			}

			xhr.onreadystatechange = function(e) {
				if (xhr.readyState == 4 && xhr.status == 200) {
				    var response = e.target.response;
				    //var response = JSON.parse(xhr.responseText);
				 	//console.log(response);
				    setupPatentAnnotations(response);
				} else  if (xhr.status != 200) {
					AjaxError3(xhr);
				}
			};
			xhr.send(formData);
		} else {
			// request for extraction, returning TEI result
			var xhr = new XMLHttpRequest();
			var url = $('#gbdForm3').attr('action');
			xhr.responseType = 'xml';
			xhr.onreadystatechange = function(e) {
				if (xhr.readyState == 4 && xhr.status == 200) {
				    var response = e.target.response;
				    //var response = JSON.parse(xhr.responseText);
				 	//console.log(response);
				    SubmitSuccesful3(response);
				} else if (xhr.status != 200) {
					AjaxError3(xhr);
				}
			};

			if (document.getElementById("input3").files && 
				document.getElementById("input3").files.length >0 &&
				!$('#textInputDiv3').is(":visible")) {
				var formData = new FormData();

				var url = $('#gbdForm3').attr('action');

				var formData = new FormData();
				formData.append('input', document.getElementById("input3").files[0]);
				
				if ($("#consolidate3").is(":checked"))	
					formData.append('consolidateCitations', 1);
				else
					formData.append('consolidateCitations', 0);
				
				xhr.open('POST', url, true);
				ShowRequest3();

				xhr.send(formData);
			} else if ($('#textInputDiv3').is(":visible")) {
				var params = 'text='+encodeURIComponent($("#textInputArea3").val());
				if ($("#consolidate3").is(":checked"))	
					params += '&consolidateCitations=1';
				else 
					params += '&consolidateCitations=0';

				xhr.open('POST', url, true);
				xhr.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
				ShowRequest3();

				xhr.send(params);
			}
		}
	}

	function setupAnnotations(response) {
		// we must check/wait that the corresponding PDF page is rendered at this point
		$('#infoResult2').html('');
		var json = response;
		var pageInfo = json.pages;

		var page_height = 0.0;
		var page_width = 0.0;

		var refBibs = json.refBibs;
		var mapRefBibs = {};
		if (refBibs) {
			for(var n in refBibs) {
				var annotation = refBibs[n];
				var theId = annotation.id;
				var theUrl = annotation.url;
				var pos = annotation.pos;
				if (pos)
					mapRefBibs[theId] = annotation;
				//for (var m in pos) {
				pos.forEach(function(thePos, m) {
					//var thePos = pos[m];
					// get page information for the annotation
					var pageNumber = thePos.p;
					if (pageInfo[pageNumber-1]) {
						page_height = pageInfo[pageNumber-1].page_height;
						page_width = pageInfo[pageNumber-1].page_width;
					}
					annotateBib(true, theId, thePos, theUrl, page_height, page_width, null);
				});
			}
		}

		// we need the above mapRefBibs structure to be created to perform the ref. markers analysis
		var refMarkers = json.refMarkers;
		if (refMarkers) {
			//for(var n in refMarkers) {
			refMarkers.forEach(function(annotation, n) {
				//var annotation = refMarkers[n];
				var theId = annotation.id;
				if (!theId)
                    return;
				// we take the first and last positions
				var targetBib = mapRefBibs[theId];
				if (targetBib) {
					var theBibPos = {};
					var pos = targetBib.pos;
					//if (pos && (pos.length > 0)) {
					var theFirstPos = pos[0];
					var theLastPos = pos[pos.length-1];
					theBibPos.p = theFirstPos.p;
					theBibPos.w = Math.max(theFirstPos.w, theLastPos.w);
					theBibPos.h = Math.max(Math.abs(theLastPos.y - theFirstPos.y), theFirstPos.h) + Math.max(theFirstPos.h, theLastPos.h);
					theBibPos.x = Math.min(theFirstPos.x, theLastPos.x);
					theBibPos.y = Math.min(theFirstPos.y, theLastPos.y);
					var pageNumber = theBibPos.p;
					if (pageInfo[pageNumber-1]) {
						page_height = pageInfo[pageNumber-1].page_height;
						page_width = pageInfo[pageNumber-1].page_width;
					}
					annotateBib(false, theId, annotation, null, page_height, page_width, theBibPos);
					//}
				} else {
					var pageNumber = annotation.p;
					if (pageInfo[pageNumber-1]) {
						page_height = pageInfo[pageNumber-1].page_height;
						page_width = pageInfo[pageNumber-1].page_width;
					}
					annotateBib(false, theId, annotation, null, page_height, page_width, null);
				}
			});
		}
	}

	function annotateBib(bib, theId, thePos, url, page_height, page_width, theBibPos) {
		var page = thePos.p;
		var pageDiv = $('#page-'+page);
		var canvas = pageDiv.children('canvas').eq(0);;

		var canvasHeight = canvas.height();
		var canvasWidth = canvas.width();
		var scale_x = canvasHeight / page_height;
		var scale_y = canvasWidth / page_width;

		var x = thePos.x * scale_x;
		var y = thePos.y * scale_y;
		var width = thePos.w * scale_x;
		var height = thePos.h * scale_y;

//console.log('annotate: ' + page + " " + x + " " + y + " " + width + " " + height);
//console.log('location: ' + canvasHeight + " " + canvasWidth);
//console.log('location: ' + page_height + " " + page_width);
		//make clickable the area
		var element = document.createElement("a");
		var attributes = "display:block; width:"+width+"px; height:"+height+"px; position:absolute; top:"+y+"px; left:"+x+"px;";

		if (bib) {
			// this is a bibliographical reference
			// we draw a line
			if (url) {
				element.setAttribute("style", attributes + "border:2px; border-style:none none solid none; border-color: blue;");
				element.setAttribute("href", url);
				element.setAttribute("target", "_blank");
			}
			else
				element.setAttribute("style", attributes + "border:1px; border-style:none none dotted none; border-color: gray;");
			element.setAttribute("id", theId);
		} else {
			// this is a reference marker
			// we draw a box
			element.setAttribute("style", attributes + "border:1px solid; border-color: blue;");
			// the link here goes to the bibliographical reference
			if (theId) {
				element.onclick = function() {
					goToByScroll(theId);
				};
			}
			// we need the area where the actual target bibliographical reference is
			if (theBibPos) {
				element.setAttribute("data-toggle", "popover");
				element.setAttribute("data-placement", "top");
				element.setAttribute("data-content", "content");
				element.setAttribute("data-trigger", "hover");
				var newWidth = theBibPos.w * scale_x;
				var newHeight = theBibPos.h * scale_y;
				var newImg = getImagePortion(theBibPos.p, newWidth, newHeight, theBibPos.x * scale_x, theBibPos.y * scale_y);
				$(element).popover({
					content:  function () {
						return '<img src=\"'+ newImg + '\" style=\"width:100%\" />';
						//return '<img src=\"'+ newImg + '\" />';
					},
					html: true,
					container: 'body'
					//width: newWidth + 'px',
					//height: newHeight + 'px'
//					container: canvas,
					//width: '600px',
					//height: '100px'
    			});
			}
		}
		pageDiv.append(element);
	}

	/* jquery-based movement to an anchor, without modifying the displayed url and a bit smoother */
	function goToByScroll(id) {
    	$('html,body').animate({scrollTop: $("#"+id).offset().top},'fast');
	}

	/* croping an area from a canvas */
	function getImagePortion(page, width, height, x, y) {
//console.log("page: " + page + ", width: " + width + ", height: " + height + ", x: " + x + ", y: " + y);
		// get the page div
		var pageDiv = $('#page-'+page);
//console.log(page);
		// get the source canvas
		var canvas = pageDiv.children('canvas')[0];
		// the destination canvas
		var tnCanvas = document.createElement('canvas');
 		var tnCanvasContext = tnCanvas.getContext('2d');
 		tnCanvas.width = width;
    	tnCanvas.height = height;
		tnCanvasContext.drawImage(canvas, x , y, width, height, 0, 0, width, height);
 		return tnCanvas.toDataURL();
	}

	function SubmitSuccesful3(responseText, statusText, xhr) {
		var display = "<pre class='prettyprint lang-xml' id='xmlCode'>";
		var testStr = vkbeautify.xml(responseText);
        teiPatentToDownload = responseText;
		display += htmll(testStr);
		display += "</pre>";
		$('#requestResult3').html(display);
		window.prettyPrint && prettyPrint();
		$('#requestResult3').show();
        $("#btn_download3").show();
	}

	function setupPatentAnnotations(response) {
		// we must check/wait that the corresponding PDF page is rendered at this point

		var json = response;
		var pageInfo = json.pages;

		var page_height = 0.0;
		var page_width = 0.0;

		var patents = json.patents;
		if (patents) {
			for(var n in patents) {
				var annotation = patents[n];
				var pos = annotation.pos;
				var theUrl = null;
				if (annotation.url && annotation.url.espacenet)
					theUrl = annotation.url.espacenet;
				else if (annotation.url && annotation.url.epoline)
					theUrl = annotation.url.epoline;
				pos.forEach(function(thePos, m) {
					// get page information for the annotation
					var pageNumber = thePos.p;
					if (pageInfo[pageNumber-1]) {
						page_height = pageInfo[pageNumber-1].page_height;
						page_width = pageInfo[pageNumber-1].page_width;
					}
					annotatePatentBib(true, thePos, theUrl, page_height, page_width);
				});
			}
		}

		var refBibs = json.articles;
		if (refBibs) {
			for(var n in refBibs) {
				var annotation = refBibs[n];
				//var theId = annotation.id;
				var theUrl = null;
				var pos = annotation.pos;
				//if (pos)
				//	mapRefBibs[theId] = annotation;
				//for (var m in pos) {
				pos.forEach(function(thePos, m) {
					//var thePos = pos[m];
					// get page information for the annotation
					var pageNumber = thePos.p;
					if (pageInfo[pageNumber-1]) {
						page_height = pageInfo[pageNumber-1].page_height;
						page_width = pageInfo[pageNumber-1].page_width;
					}
					annotatePatentBib(false, thePos, theUrl, page_height);
				});
			}
		}
	}

	function annotatePatentBib(isPatent, thePos, url, page_height, page_width, theBibPos) {
		var page = thePos.p;
		var pageDiv = $('#page-'+page);
		var canvas = pageDiv.children('canvas').eq(0);

		var canvasHeight = canvas.height();
		var canvasWidth = canvas.width();
		var scale_y = canvasHeight / page_height;
		var scale_x = canvasWidth / page_width;

		var x = thePos.x * scale_x;
		var y = thePos.y * scale_y;
		var width = thePos.w * scale_x;
		var height = thePos.h * scale_y;

//console.log('annotate: ' + page + " " + x + " " + y + " " + width + " " + height);
//console.log('location: ' + canvasHeight + " " + canvasWidth);
//console.log('location: ' + page_height + " " + page_width);
		//make clickable the area
		var element = document.createElement("a");
		var attributes = "display:block; width:"+width+"px; height:"+height+"px; position:absolute; top:"+y+"px; left:"+x+"px;";

		if (patent) {
			// this is a patent reference
			// we draw a line
			if (url) {
				element.setAttribute("style", attributes + "border:2px; border-style:none none solid none; border-color: blue;");
				element.setAttribute("href", url);
				element.setAttribute("target", "_blank");
			}
			else
				element.setAttribute("style", attributes + "border:1px; border-style:none none dotted none; border-color: gray;");
		} else {
			// this is a NPL bibliographical reference
			// we draw a box
			element.setAttribute("style", attributes + "border:1px solid; border-color: blue;");

			/*element.setAttribute("data-toggle", "popover");
			element.setAttribute("data-placement", "top");
			element.setAttribute("data-content", "content");
			element.setAttribute("data-trigger", "hover");

			$(element).popover({
				content:  'content',
				html: true,
				container: 'body'
				//width: newWidth + 'px',
				//height: newHeight + 'px'
//					container: canvas,
				//width: '600px',
				//height: '100px'
			});*/

		}
		pageDiv.append(element);
	}

	$(document).ready(function() {
	    $(document).on('shown', '#xmlCode', function(event) {
	        prettyPrint();
	    });
	});

	function processChange() {
		var selected = $('#selectedService option:selected').attr('value');
		if (block == 1)
			selected = $('#selectedService2 option:selected').attr('value');
		else if (block == 2)
			selected = $('#selectedService3 option:selected').attr('value');

		if (selected == 'processHeaderDocument') {
			createInputFile(selected);
			$('#consolidateBlock1').show();
			$('#consolidateBlock2').hide();
			setBaseUrl('processHeaderDocument');
		}
		else if (selected == 'processFulltextDocument') {
			createInputFile(selected);
			$('#consolidateBlock1').show();
			$('#consolidateBlock2').show();
			setBaseUrl('processFulltextDocument');
		}
		else if (selected == 'processDate') {
			createInputTextArea('date');
			$('#consolidateBlock1').hide();
			$('#consolidateBlock2').hide();
			setBaseUrl('processDate');
		}
		else if (selected == 'processHeaderNames') {
			createInputTextArea('names');
			$('#consolidateBlock1').hide();
			$('#consolidateBlock2').hide();
			setBaseUrl('processHeaderNames');
		}
		else if (selected == 'processCitationNames') {
			createInputTextArea('names');
			$('#consolidateBlock1').hide();
			$('#consolidateBlock2').hide();
			setBaseUrl('processCitationNames');
		}
		else if (selected == 'processReferences') {
			createInputFile(selected);
			$('#consolidateBlock1').hide();
			$('#consolidateBlock2').show();
			setBaseUrl('processReferences');
		}
		else if (selected == 'processAffiliations') {
			createInputTextArea('affiliations');
			$('#consolidateBlock1').hide();
			$('#consolidateBlock2').hide();
			setBaseUrl('processAffiliations');
		}
		else if (selected == 'processCitation') {
			createInputTextArea('citations');
			$('#consolidateBlock1').hide();
			$('#consolidateBlock2').show();
			setBaseUrl('processCitation');
		}
		/*else if (selected == 'processCitationPatentTEI') {
			createInputFile3(selected);
			$('#consolidateBlock3').show();
			setBaseUrl('processCitationPatentTEI');
		}*/
		else if (selected == 'processCitationPatentST36') {
			createInputFile3(selected);
			$('#consolidateBlock3').show();
			setBaseUrl('processCitationPatentST36');
		}
		else if (selected == 'processCitationPatentPDF') {
			createInputFile3(selected);
			$('#consolidateBlock3').show();
			setBaseUrl('processCitationPatentPDF');
		}
		else if (selected == 'processCitationPatentTXT') {
			createInputTextArea3('text');
			$('#consolidateBlock3').show();
			setBaseUrl('processCitationPatentTXT');
		}
		else if (selected == 'referenceAnnotations') {
			createInputFile2(selected);
			$('#consolidateBlock').show();
			setBaseUrl('referenceAnnotations');
		}
		else if (selected == 'annotatePDF') {
			createInputFile2(selected);
			$('#consolidateBlock').show();
			setBaseUrl('annotatePDF');
		}
		else if (selected == 'citationPatentAnnotations') {
			createInputFile3(selected);
			$('#consolidateBlock3').show();
			setBaseUrl('citationPatentAnnotations');
		}
	}

	function createInputFile(selected) {
		//$('#label').html('&nbsp;');
		$('#textInputDiv').hide();
		$('#fileInputDiv').show();

		$('#gbdForm').attr('enctype', 'multipart/form-data');
		$('#gbdForm').attr('method', 'post');
	}

	function createInputFile2(selected) {
		//$('#label').html('&nbsp;');
		$('#textInputDiv2').hide();
		$('#fileInputDiv2').show();

		$('#gbdForm2').attr('enctype', 'multipart/form-data');
		$('#gbdForm2').attr('method', 'post');
	}

	function createInputFile3(selected) {
		//$('#label').html('&nbsp;');
		$('#textInputDiv3').hide();
		$('#fileInputDiv3').show();

		$('#gbdForm3').attr('enctype', 'multipart/form-data');
		$('#gbdForm3').attr('method', 'post');
	}

	function createInputTextArea(nameInput) {
		//$('#label').html('&nbsp;');
		$('#fileInputDiv').hide();
		//$('#input').remove();

		//$('#field').html('<table><tr><td><textarea class="span7" rows="5" id="input" name="'+nameInput+'" /></td>'+
		//"<td><span style='padding-left:20px;'>&nbsp;</span></td></tr></table>");
		$('#textInputArea').attr('name', nameInput);
		$('#textInputDiv').show();

		$('#gbdForm').attr('enctype', '');
		$('#gbdForm').attr('method', 'post');
	}

	function createInputTextArea3(nameInput) {
		//$('#label').html('&nbsp;');
		$('#fileInputDiv3').hide();

		$('#textInputArea3').attr('name', nameInput);
		$('#textInputDiv3').show();

		$('#gbdForm3').attr('enctype', '');
		$('#gbdForm3').attr('method', 'post');
	}

	function download(){
        var name ="export";
		if ((document.getElementById("input").files[0].type == 'application/pdf') ||
            (document.getElementById("input").files[0].name.endsWith(".pdf")) ||
            (document.getElementById("input").files[0].name.endsWith(".PDF"))) {
             name = document.getElementById("input").files[0].name;
        }
		var fileName = name + ".tei.xml";
	    var a = document.createElement("a");


	    var file = new Blob([teiToDownload], {type: 'application/xml'});
	    var fileURL = URL.createObjectURL(file);
	    a.href = fileURL;
	    a.download = fileName;

	    document.body.appendChild(a);

	    $(a).ready(function() {
			a.click();
			return true;
		});


		// old method to download but with well formed xm but not beautified
	    /*var a = document.body.appendChild(
	        document.createElement("a")
	    );
	    a.download = "export.xml";
	    var xmlData = $.parseXML(teiToDownload);

	    if (window.ActiveXObject){
	        var xmlString = xmlData.xml;
	    } else {
	        var xmlString = (new XMLSerializer()).serializeToString(xmlData);
	    }
	    a.href = "data:text/xml," + xmlString; // Grab the HTML
	    a.click(); // Trigger a click on the element*/
	}

	function downloadPatent() {
        var name = "export";
        if ((document.getElementById("input3").files[0].type == 'application/pdf') ||
            (document.getElementById("input3").files[0].name.endsWith(".pdf")) ||
            (document.getElementById("input3").files[0].name.endsWith(".PDF"))) {
            name = document.getElementById("input3").files[0].name;
        }
        var fileName = name + ".tei.xml";
        var a = document.createElement("a");


        var file = new Blob([teiPatentToDownload], {type: 'application/xml'});
        var fileURL = URL.createObjectURL(file);
        a.href = fileURL;
        a.download = fileName;

        document.body.appendChild(a);

        $(a).ready(function () {
            a.click();
            return true;
        });

    }
    })(jQuery);


function downloadVisibilty(){
    $("#btn_download").hide();
}
function downloadVisibilty3(){
    $("#btn_download3").hide();
}