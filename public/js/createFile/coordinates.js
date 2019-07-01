const PDFExtract = require('pdf.js-extract').PDFExtract;
const fs = require('fs');
var stringSimilarity = require('string-similarity');
const hummus = require('hummus');
const PDFRStreamForBuffer = require('./PDFRStreamForBuffer');
const { work } = require('./workpdf');
const pdfExtract = new PDFExtract();
const options = {}; /* see below */
var annotations = []
var tempAnnotations = [];
var postTempAnnotations = [];
var highlightThis = "";
var highlightThisArr = [];

function createPdf(annotationsToHighlight, path, outpath){
    let trying = fs.readFileSync(path);
    const inStream = new PDFRStreamForBuffer(trying);
    const mystream = new hummus.PDFWStreamForFile(outpath)

    work(annotationsToHighlight, inStream, mystream);
}

function pushToArray(extraction, inputArray) {
    inputArray.forEach((element) => {
        annotations.push({
            "page": parseInt(element.pageFound),
            "position": [element.x, 800 - element.y, element.x + element.width, 800 - element.y - 2 * element.height / 2],
            "content": element.str,
            "color": [255, 255, 0]
        });
    });
}

function findWord(extraction, word) {
 extraction.pages.forEach((page, pageindex) => {
     page.content.forEach((theelement, thelementindex) => {
     theelement.str = theelement.str.replace(/[^a-zA-Z0-9.?!,% ]/g, "");
        const validelement = (theelement.str.length > 0);
        if (validelement) {
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
    possibilities.forEach((element) => {
        const elementToPush = extraction.pages[element.toString().substring(0, element.toString().indexOf("."))].content[element.toString().substring(element.toString().indexOf(".") + 1)]
        elementToPush.pageFound = element.toString().substring(0, element.toString().indexOf("."));
        elementToPush.elementFound = element.toString().substring(element.toString().indexOf(".") + 1);
        uniquePossibilities.push(elementToPush);
    });
    var matches = stringSimilarity.findBestMatch(highlightThis, uniquePossibilities.map(function(ob) {
        return ob.str;
    }));
    postTempAnnotations.push(uniquePossibilities[matches.bestMatchIndex]);
    return uniquePossibilities[matches.bestMatchIndex];
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
    if(stringSimilarity.compareTwoStrings(extraction.pages[firstPrevPage].content[firstPrevLine].str, highlightThis) > 0.47) {
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
    console.log(stringSimilarity.compareTwoStrings(extraction.pages[firstAfterPage].content[firstAfterLine].str, highlightThis))
    if(stringSimilarity.compareTwoStrings(extraction.pages[firstAfterPage].content[firstAfterLine].str, highlightThis) > 0.47) {
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
    const index = annotations[annotations.length - 1].content.indexOf(highlightThisArr[highlightThisArr.length - 1]) + highlightThisArr[highlightThisArr.length - 1].length;
    const totalIndex = annotations[annotations.length - 1].content.length;
    const proportion = (index + 0.0) / totalIndex
    return annotations[annotations.length - 1].position[0] + (proportion * (annotations[annotations.length - 1].position[2] - annotations[annotations.length - 1].position[0]))
}

function trimBeforePeriod() {
    const index = annotations[0].content.indexOf(highlightThisArr[0]);
    const totalIndex = annotations[0].content.length;
    let proportion = (index + 0.0) / totalIndex
    console.log(proportion)
    if (proportion < 0) {
    proportion = 0;
    }
    return (proportion * (annotations[0].position[2] - annotations[0].position[0]));
}


function findTheCoord(arrayToHighlight, extraction, path, outpath) {
	arrayToHighlight = Array.from(arrayToHighlight)
	console.log(typeof(arrayToHighlight))
	arrayToHighlight.forEach((strToHighlight, strindex) => {
		tempAnnotations = [];
		postTempAnnotations = [];
		highlightThis = strToHighlight.replace(/[^a-zA-Z0-9.!?,% ]/g, "");
		highlightThisArr = highlightThis.split(" ");
		let wordsToSearch = [];
		highlightThisArr.forEach((word) => {
			if (word.length > 3) {
				wordsToSearch.push(word);
			}
		});
		if (wordsToSearch.length >= 4) {
			findWord(extraction, wordsToSearch[0]);
			findWord(extraction, wordsToSearch[1]);
			findWord(extraction, wordsToSearch[wordsToSearch.length - 1]);
			findWord(extraction, wordsToSearch[wordsToSearch.length - 2]);
		} //need an else clause for very short, simple sentence
		sortArray(tempAnnotations);
		let originalSeedBlock = checkArray(extraction, tempAnnotations);
		fillRangeBefore(extraction, originalSeedBlock);
		fillRangeAfter(extraction, originalSeedBlock);
		pushToArray(extraction, postTempAnnotations);
		console.log(annotations);
		const trimBefore = trimBeforePeriod();
        const trimAfter = trimAfterPeriod();
        annotations[0].position[0] += trimBefore;
        annotations[annotations.length - 1].position[2] = trimAfter;
        console.log(annotations)
	});
	createPdf(annotations, path, outpath);
}

module.exports = {
	extractor: function (highlightThis, path, outpath) {
		pdfExtract.extract(path, options, (err, data) => {
    		annotations = []
    		if (err) return console.log(err);
    		console.log(data)
    		findTheCoord(highlightThis, data, path, outpath);
	   });
	}
};

//extractor(["Non-Traditional Narrative Elements in ​Moby Dick","In addition, Pip is a character that Melville uses for dramatization."],'/Users/etashguha/Downloads/Etash.pdf', './mycasdfn.pdf');