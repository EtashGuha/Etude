const PDFExtract = require('pdf.js-extract').PDFExtract;
const fs = require('fs');
var stringSimilarity = require('string-similarity');
const hummus = require('hummus');
const PDFRStreamForBuffer = require('./PDFRStreamForBuffer');
const {
    work
} = require('./workpdf');
const pdfExtract = new PDFExtract();
const options = {}; /* see below */
var annotations = []
var tempAnnotations = [];
var postTempAnnotations = [];
var highlightThis = "";
var highlightThisArr = [];
var etudefilepath = "";

function createPdf(annotationsToHighlight, path, outpath) {
    let trying = fs.readFileSync(path);
    const inStream = new PDFRStreamForBuffer(trying);
    const mystream = new hummus.PDFWStreamForFile(outpath)

    work(annotationsToHighlight, inStream, mystream);
    fs.writeFile(etudefilepath + "/tmp/object.json", JSON.stringify(annotationsToHighlight), (err) => {
        if (err) {
            console.error(err);
            return;
        };
        console.log(etudefilepath + "/tmp/object.json")
        console.log("File has been created");
    });
}

function pushToArray(extraction, inputArray) {
    //console.log(extraction.pages[0].pageInfo.height);
    inputArray.forEach((element) => {
        annotations.push({
            "page": parseInt(element.pageFound),
            "position": [element.x, extraction.pages[0].pageInfo.height - element.y, element.x + element.width, extraction.pages[0].pageInfo.height - element.y + 2 * element.height / 2],
            "content": element.str,
            "color": [255, 255, 0]
        });
    });
}

function findWord(extraction, word) {
    //console.log(word);
    extraction.pages.forEach((page, pageindex) => {
        //console.log(page);
        page.content.forEach((theelement, thelementindex) => {
            theelement.str = theelement.str.replace(/[^a-zA-Z0-9.?!,% ]/g, "");
            theelement.str = theelement.str.replace(/\s\s+/g, ' ');
            const validelement = (theelement.str.length > 0);
            if (validelement) {
                //console.log(theelement.str);
                if (theelement.str.indexOf(word) !== -1) {
                    tempAnnotations.push([pageindex, thelementindex]);
                }
            }
        });
    });
}

function sortArray(arr) {
    arr.sort(function(a, b) {
        if (a[0] < b[0]) {
            return -1;
        } else if (a[0] > b[0]) {
            return 1;
        } else {
            if (a[1] < b[1]) {
                return -1;
            } else if (a[1] > b[1]) {
                return 1;
            }
        }
    });
}

function checkArray(extraction, arr) {
    let possibilities = new Set();
    arr.forEach((item) => {
        possibilities.add(item[0] + "." + item[1]);
    });
    let uniquePossibilities = [];
    //console.log(possibilities);
    possibilities.forEach((element) => {
        const elementToPush = extraction.pages[element.toString().substring(0, element.toString().indexOf("."))].content[element.toString().substring(element.toString().indexOf(".") + 1)]
        //console.log(elementToPush);
        elementToPush.pageFound = element.toString().substring(0, element.toString().indexOf("."));
        elementToPush.elementFound = element.toString().substring(element.toString().indexOf(".") + 1);
        uniquePossibilities.push(elementToPush);
    });
    //console.log(highlightThis);
    //console.log(uniquePossibilities);
    if (uniquePossibilities !== undefined && uniquePossibilities.length !== 0) {
        var matches = stringSimilarity.findBestMatch(highlightThis, uniquePossibilities.map(function(ob) {
            return ob.str;
        }));
        postTempAnnotations.push(uniquePossibilities[matches.bestMatchIndex]);
        //console.log(uniquePossibilities[matches.bestMatchIndex]);
        return uniquePossibilities[matches.bestMatchIndex];
    } else {
        return null;
    }
}


function fillRangeBefore(extraction, theObject) {
    let firstPrevPage = parseInt(theObject.pageFound);
    let firstPrevLine = parseInt(theObject.elementFound) - 1;
    try {
        while (extraction.pages[firstPrevPage].content[firstPrevLine].str.length < 2) {
            firstPrevLine -= 1;
        }
    } catch (err) {
        firstPrevPage -= 1;
        firstPrevLine = extraction.pages[firstPrevPage].content.length - 1;
        while (extraction.pages[firstPrevPage].content[firstPrevLine].str.length < 2) {
            firstPrevLine -= 1;
        }
    }
    if (stringSimilarity.compareTwoStrings(extraction.pages[firstPrevPage].content[firstPrevLine].str, highlightThis) > 0.47) {
        let afterPush = extraction.pages[firstPrevPage].content[firstPrevLine];
        afterPush.pageFound = firstPrevPage;
        afterPush.elementFound = firstPrevLine;
        postTempAnnotations.unshift(afterPush);
        fillRangeBefore(extraction, afterPush);
    } else {
        if (postTempAnnotations[0].str.indexOf(highlightThisArr[0]) === -1) {
            console.log('Mandate of Heaven')
            let afterPush = extraction.pages[firstPrevPage].content[firstPrevLine];
            afterPush.pageFound = firstPrevPage;
            afterPush.elementFound = firstPrevLine;
            postTempAnnotations.unshift(afterPush);
        }
    }
}

function fillRangeAfter(extraction, theObject) {
    let firstAfterPage = parseInt(theObject.pageFound);
    let firstAfterLine = 1 + parseInt(theObject.elementFound);
    try {
        while (extraction.pages[firstAfterPage].content[firstAfterLine].str.length < 2) {
            firstAfterLine += 1;
        }
    } catch (err) {
        firstAfterPage += 1;
        firstAfterLine = 0;
        while (extraction.pages[firstAfterPage].content[firstAfterLine].str.length < 2) {
            firstAfterLine += 1;
        }
    }
    //console.log(stringSimilarity.compareTwoStrings(extraction.pages[firstAfterPage].content[firstAfterLine].str, highlightThis))
    if (stringSimilarity.compareTwoStrings(extraction.pages[firstAfterPage].content[firstAfterLine].str, highlightThis) > 0.47) {
        let afterPush = extraction.pages[firstAfterPage].content[firstAfterLine];
        afterPush.pageFound = firstAfterPage;
        afterPush.elementFound = firstAfterLine;
        postTempAnnotations.push(afterPush);
        fillRangeAfter(extraction, afterPush);
    } else {
        if (postTempAnnotations[postTempAnnotations.length - 1].str.indexOf(highlightThisArr[highlightThisArr.length - 1]) === -1) {
            console.log('Mandate of Heaven')
            let afterPush = extraction.pages[firstAfterPage].content[firstAfterLine];
            afterPush.pageFound = firstAfterPage;
            afterPush.elementFound = firstAfterLine;
            postTempAnnotations.push(afterPush);
        }
    }
}

function trimAfterPeriod() {
    console.log(postTempAnnotations[postTempAnnotations.length - 1].str + highlightThisArr[highlightThisArr.length - 1])
    if (postTempAnnotations[postTempAnnotations.length - 1].str.indexOf(highlightThisArr[highlightThisArr.length - 1]) !== -1) {
        const index = postTempAnnotations[postTempAnnotations.length - 1].str.indexOf(highlightThisArr[highlightThisArr.length - 1]) + highlightThisArr[highlightThisArr.length - 1].length;
        const totalIndex = postTempAnnotations[postTempAnnotations.length - 1].str.length;
        const proportion = (index + 0.0) / totalIndex
        console.log(proportion);
        postTempAnnotations[postTempAnnotations.length - 1].oldWidth = postTempAnnotations[postTempAnnotations.length - 1].width
        postTempAnnotations[postTempAnnotations.length - 1].width = (proportion * (postTempAnnotations[postTempAnnotations.length - 1].width))
        console.log(postTempAnnotations[postTempAnnotations.length - 1].width + postTempAnnotations[postTempAnnotations.length - 1].x)
    }
}

function trimBeforePeriod() {

    //console.log("trimming" + highlightThisArr[0] + postTempAnnotations[0].str);
    const index = postTempAnnotations[0].str.indexOf(highlightThisArr[0]);
    const totalIndex = postTempAnnotations[0].str.length;
    //console.log(index);
    //console.log(totalIndex);
    let proportion = (index + 0.0) / totalIndex
    //console.log(proportion)
    if (proportion < 0) {
        proportion = 0;
    }
    //console.log(postTempAnnotations[0].x)
    if (postTempAnnotations[0].oldWidth !== undefined) {
        postTempAnnotations[0].x = postTempAnnotations[0].x + (proportion * (postTempAnnotations[0].oldWidth));
        //  console.log(postTempAnnotations[0].x)
        if (postTempAnnotations.length !== 1) {
            postTempAnnotations[0].width = ((1 - proportion) * (postTempAnnotations[0].oldWidth))
        }
    } else {
        postTempAnnotations[0].x = postTempAnnotations[0].x + (proportion * (postTempAnnotations[0].width));
        //console.log(postTempAnnotations[0].x)
        if (postTempAnnotations.length !== 1) {
            postTempAnnotations[0].width = ((1 - proportion) * (postTempAnnotations[0].width))
        }
    }
}

function postProcess(extraction) {
    let prevElement = {
        x: -1,
        y: -1,
        str: "",
        height: 0,
        width: 0
    };
    extraction.pages.forEach((page, pageindex) => {
        page.content = page.content.filter((element, index) => {
            if (element.y === prevElement.y) {
                prevElement.str = prevElement.str + element.str;
                prevElement.width = prevElement.width + element.width;
                return false;
            } else {
                prevElement = element;
                return true;
            }
        });
    });
    //console.log(extraction);
    return extraction;
}

function findTheCoord(arrayToHighlight, extraction, path, outpath) {
    console.log(annotations.length)
    arrayToHighlight = Array.from(arrayToHighlight)
    arrayToHighlight.forEach((strToHighlight, strindex) => {
        tempAnnotations = [];
        postTempAnnotations = [];
        highlightThis = strToHighlight.replace(/[^a-zA-Z0-9.!?,% ]/g, "");
        highlightThis = highlightThis.replace(/\s\s+/g, ' ');
        highlightThisArr = highlightThis.split(" ");
        let wordsToSearch = [];
        highlightThisArr.forEach((word) => {
            if (word.length > 3) {
                wordsToSearch.push(word);
            }
        });
        extraction = postProcess(extraction);
        if (wordsToSearch.length >= 4) {
            //console.log(wordsToSearch);
            findWord(extraction, wordsToSearch[0]);
            findWord(extraction, wordsToSearch[1]);
            findWord(extraction, wordsToSearch[wordsToSearch.length - 1]);
            findWord(extraction, wordsToSearch[wordsToSearch.length - 2]);
        } //need an else clause for very short, simple sentence
        //console.log(tempAnnotations);
        sortArray(tempAnnotations);
        let originalSeedBlock = checkArray(extraction, tempAnnotations);
        if (originalSeedBlock !== null) {
            fillRangeBefore(extraction, originalSeedBlock);
            fillRangeAfter(extraction, originalSeedBlock);
            trimBeforePeriod();
            trimAfterPeriod();

            pushToArray(extraction, postTempAnnotations);
            console.log(annotations.length)
            //console.log(postTempAnnotations)

            // console.log(annotations)
            // console.log(highlightThisArr);
            //trimBeforePeriod(annotations);
        }
    });
    //console.log(annotations);
    createPdf(annotations, path, outpath);
}

module.exports = {
    extractor: function(highlightThis, path, outpath, filepathForEtude) {
        pdfExtract.extract(path, options, (err, data) => {
            annotations = []
            if (err) return console.log(err);
            etudefilepath = filepathForEtude;
            findTheCoord(highlightThis, data, path, outpath);
        });
    }
};

//extractor(["Non-Traditional Narrative Elements in ​Moby Dick","In addition, Pip is a character that Melville uses for dramatization."],'/Users/etashguha/Downloads/Etash.pdf', './mycasdfn.pdf');