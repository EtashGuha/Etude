const lemmatizer = require("lemmatizer")
var thesaurus = require("thesaurus");
var stringSimilarity = require('string-similarity');
var HashMap = require('hashmap');
const fs = require('fs');
var spellChecker = require('spellchecker')
const {
	MinHeap,
	MaxHeap
} = require('@datastructures-js/heap');
const minHeap = new MinHeap();

var map = new HashMap();

var text = fs.readFileSync("/Users/etashguha/Documents/etude/example.txt", 'utf8')
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
var question = "What is the point of view of the narrative?"

function keyword(s) {
	var re = new RegExp('\\b(' + stopwords.join('|') + ')\\b', 'g');
	return (s || '').replace(re, '').replace(/[ ]{2,}/, ' ');
}

function tokenize(str) {
	var words = str.split(/\W+/).filter(function(token) {
		token = token.toLowerCase();
		token = token.replace(/[^a-z ]/gi, '')
		return token.length >= 2 && stopwords.indexOf(token) == -1;
	});
	return new Set(words)
}

function lemmatizeSet(inSet) {
	var lemmatized = new Set()
	inSet.forEach((item) => {
		lemmatized.add(item)
		lemmatized.add(lemmatizer.lemmatizer(item))
	})
	return lemmatized;
}

function includes(stringToSearch, substr) {
	var substrLength = substr.length
	var strToSearLength = stringToSearch.length
	for(var i = 0; i < strToSearLength - substrLength; i++){
		if(stringToSearch.substring(i, i + substrLength) == substr){
			return true;
		}
	}

	return false;
}


async function getAnswer(question, text){
	var textArray = text.toLowerCase().match(/[^\.!\?]+[\.!\?]+/g)
	// console.log(textArray)
	question = tokenize(keyword(question.toLowerCase()))

	question.forEach((item) => {
		map.set(lemmatizer.lemmatizer(item), thesaurus.find(lemmatizer.lemmatizer(item)))
		map.set(item, thesaurus.find(item))
	})
 	var pastSentences = new Set()
	for (var i = textArray.length - 1; i >= 0; i--) {
		console.log(i)
		var currSentence = tokenize(keyword(textArray[i]))
		// console.log(currSentence)
		if(currSentence.size < 1){
			console.log("dont care")
			continue
		}
		if(pastSentences.has(currSentence)){
			console.log("seen already")
			continue
		}
		pastSentences.add(currSentence)

		var matchList = new Set(
			[...question].filter(x => currSentence.has(x)));

		var qlessc = new Set(
			[...question].filter(x => !currSentence.has(x)));

		var clessq = new Set(
			[...currSentence].filter(x => !question.has(x)));

		qlessc = lemmatizeSet(qlessc)
		clessq = lemmatizeSet(clessq)
		var sharedSize = matchList.size
		var noMatchList = []
		
		qlessc.forEach((item) => {
			var synonymList = map.get(item)
			synonymList.push(item)
			var included = false
			for (var m = synonymList.length - 1; m >= 0; m--) {
				var hasThisSynonym = true
				var synonym = synonymList[m]

				var splitSynonym = synonym.split(" ")
				for (var j = splitSynonym.length - 1; j >= 0; j--) {
					if (!clessq.has(splitSynonym[j])) {
						hasThisSynonym = false;
						break;
					}
				}
				if(hasThisSynonym){
					for (var g = splitSynonym.length - 1; g >= 0; g--) {
						clessq.delete(splitSynonym[g])
					}
					included = true;
					break;
				}
			}
			if (included) {
				sharedSize += 1;
			} else {
				noMatchList.push(synonymList)
			}
		});
		clessq.forEach((item) => {
			if(!spellChecker.isMisspelled(item)){
				clessq.delete(item)
			}
		})

		var sumOfMatches = 0;

		var numTermsMatching = 0
		if(clessq.size > 0){
			console.log(clessq)
			for (var m = noMatchList.length - 1; m >= 0; m--) {
				var bestMatchForEachTerm = 0
				for (var j = noMatchList[m].length - 1; j >= 0; j--) {
					clessq.forEach((item) => {
						if(includes(item, noMatchList[m][j])) {
							bestMatchForEachTerm = 1
						}
					})
					if(bestMatchForEachTerm == 1){
						break;
					}
				}
				numTermsMatching += bestMatchForEachTerm
			}
		} else {
			console.log('none isMisspelled')
		}
		

		var rating = (sharedSize + numTermsMatching) / (question.size)

		if(isNaN(rating)){
			continue;
		}
		
		if(minHeap.size() <= 8) {
			minHeap.insert(rating, textArray[i])
		} else if(rating > minHeap.root().getKey()) {
			minHeap.insert(rating, textArray[i])
			minHeap.extractRoot()
		}
	}
	console.log(minHeap.serialize())
}

getAnswer(question, text)