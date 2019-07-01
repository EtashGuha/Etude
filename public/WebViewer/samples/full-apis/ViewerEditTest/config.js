(function() {
  var refreshSVG = function(color) {
    return '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="' + color + '"><path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/><path d="M0 0h24v24H0z" fill="none"/></svg>';
  };

  $(document).on('documentLoaded', function() {
    PDFNet.initialize().then(function() {
      var doc = readerControl.docViewer.getDocument();
      doc.getPDFDoc().then(function(pdfDoc) {
        readerControl.setHeaderItems(function(headerItems) {
          headerItems.push({
            type: 'statefulButton',
            mount: function() {
              return 'enabled';
            },
            states: {
              enabled: {
                img: refreshSVG('currentColor'),
                onClick: function(update) {
                  update('disabled');
                  runElementEditTest(pdfDoc).then(function() {
                    // re-enable our button
                    update('enabled');
                    // refresh the cache with the newly updated document
                    readerControl.docViewer.refreshAll();
                    // update viewer with new document
                    readerControl.docViewer.updateView();
                  });
                }
              },
              disabled: {
                img: refreshSVG('lightgray'),
                onClick: function() {}
              }
            }
          });

          return headerItems;
        });
      });
    });
  });

  var runElementEditTest = function(pdfDoc) {
    function* ProcessElements(reader, writer, visited) {
      yield PDFNet.startDeallocateStack();
      console.log('Processing elements');
      var element;
      var gs;
      var colorspace = yield PDFNet.ColorSpace.createDeviceRGB();
      var redColor = yield PDFNet.ColorPt.init(1, 0, 0, 0);
      var blueColor = yield PDFNet.ColorPt.init(0, 0, 1, 0);
      for (element = yield reader.next(); element !== null; element = yield reader.next()) {
        var elementType = yield element.getType();
        switch (elementType) {
          case PDFNet.Element.Type.e_image:
          case PDFNet.Element.Type.e_inline_image:
            // remove all images by skipping them
            break;
          case PDFNet.Element.Type.e_path:
            // Set all paths to red
            gs = yield element.getGState();
            gs.setFillColorSpace(colorspace);
            gs.setFillColorWithColorPt(redColor);
            // Note: since writeElement does not return an object, the yield is technically unneeded.
            // However, on a slower computer or browser writeElement may not finish before the page is
            // updated, so the yield ensures that all changes are finished before continuing.
            yield writer.writeElement(element);
            break;
          case PDFNet.Element.Type.e_text:
            // Set all text to blue
            gs = yield element.getGState();
            gs.setFillColorSpace(colorspace);
            gs.setFillColorWithColorPt(blueColor);
            // Same as above comment on writeElement
            yield writer.writeElement(element);
            break;
          case PDFNet.Element.Type.e_form:
            yield writer.writeElement(element);
            var form_obj = yield element.getXObject();
            var form_obj_num = form_obj.getObjNum();
            // if XObject not yet processed
            if (visited.indexOf(form_obj_num) === -1) {
              // Set Replacement
              var insertedObj = yield form_obj.getObjNum();
              if (_.findWhere(visited, insertedObj) == null) {
                visited.push(insertedObj);
              }
              var new_writer = yield PDFNet.ElementWriter.create();
              reader.formBegin();
              new_writer.beginOnObj(form_obj, true);
              yield* ProcessElements(reader, new_writer, visited);
              new_writer.end();
              reader.end();
              if (new_writer) {
                new_writer.destroy();
              }
            }
            break;
          default:
            yield writer.writeElement(element);
        }
      }
      yield PDFNet.endDeallocateStack();
    }

    function* main() {
      var ret = 0;
      try {
        // eslint-disable-next-line no-unused-vars
        var islocked = false;
        var doc = pdfDoc;
        doc.lock();
        islocked = true;
        doc.initSecurityHandler();

        var writer = yield PDFNet.ElementWriter.create();
        var reader = yield PDFNet.ElementReader.create();
        var visited = [];

        var pageCount = yield doc.getPageCount();

        var pageCounter = 1;
        while (pageCounter <= pageCount) {
          // This section is only required to ensure the page is available
          // for incremental download. At the moment the call to requirePage must be
          // be wrapped in this manner to avoid potential deadlocks and
          // allow other parts of the viewer to run while the page is being downloaded.
          doc.unlock();
          yield PDFNet.finishOperation();
          yield doc.requirePage(pageCounter);
          yield PDFNet.beginOperation();
          doc.lock();

          // load the page and begin processing
          var page = yield doc.getPage(pageCounter);
          var sdfObj = yield page.getSDFObj();
          var insertedObj = yield sdfObj.getObjNum();
          if (_.findWhere(visited, insertedObj) == null) {
            visited.push(insertedObj);
          }
          reader.beginOnPage(page);
          writer.beginOnPage(page, PDFNet.ElementWriter.WriteMode.e_replacement, false);
          yield* ProcessElements(reader, writer, visited);
          writer.end();
          reader.end();
          console.log('page ' + pageCounter + ' finished editing');
          pageCounter++;
        }
        console.log('Done.');
      } catch (err) {
        console.log(err.stack);
        ret = 1;
      }
      return ret;
    }

    return PDFNet.runGeneratorWithCleanup(main());
  };
})();
// eslint-disable-next-line spaced-comment
//# sourceURL=config.js