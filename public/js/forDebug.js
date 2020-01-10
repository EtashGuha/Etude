const os = require('os')
var osvers = os.platform()

var Worker = require("tiny-worker");
const lemmatizer = require("lemmatizer")
var stringSimilarity = require('string-similarity');
var HashMap = require('hashmap');
const fs = require('fs');
const spell = require('spell-checker-js')
var WordNet = require("node-wordnet")
spell.load('en')
var wordnet = new WordNet()
const {
	MinHeap,
	MaxHeap
} = require('@datastructures-js/heap');
const minHeap = new MinHeap();

// var wordnet = new natural.WordNet();

var map = new HashMap();

var text = "Who is where your dick penis question hard hard hard hard hard hard hard hard hard thebanana hard hard hard hard hard hhard hard hard see split is sally. "
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
var question = "the of banana easy easy easy easy easy easy easy split easy easy easy easy easy easy easy easy easy chicken easy easy easy easy easy easy easy easy question"

function keyword(s) {
	var re = new RegExp('\\b(' + stopwords.join('|') + ')\\b', 'g');
	return (s || '').replace(re, '').replace(/[ ]{2,}/, ' ');
}

function arrDifference(arr) {
	var differences = []
	for(var i = 0; i <= arr.length - 2; i++){
		for (var j = i + 1; j <= arr.length - 1; j++) {
			differences.push(arr[i] - arr[j])
		}
	}
	return differences
}

function alteredSigmoid(x) {
	var exponent = -1.0/3 * x
	var eterm = Math.exp(exponent)
	return 2 * eterm/(1 + eterm)
}

function tokenize(str) {
	var words = str.split(/\W+/).filter(function(token) {
		token = token.toLowerCase();
		token = token.replace(/[^a-z ]/gi, '')
		return token.length >= 2 && stopwords.indexOf(token) == -1;
	});
	return new Set(words)
}

function tokenizeArr(str){
	var words = str.split(/\W+/).filter(function(token) {
		token = token.toLowerCase();
		token = token.replace(/[^a-z ]/gi, '')
		return token;
	});
	return words
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
	originalQuestionArr = tokenizeArr(question)
	var textArray = text.toLowerCase().match(/[^\.!\?]+[\.!\?]+/g)
	question = tokenize(keyword(question.toLowerCase()))
	var questionArray = Array.from(question)
	
	var questionVector = []
	for(var i = 0; i < questionArray.length; i++){
		questionVector.push(originalQuestionArr.indexOf(questionArray[i]))
	}

	for (var k = questionArray.length - 1; k >= 0; k--) {
		var item = questionArray[k]
		var lemmatizedWord = lemmatizer.lemmatizer(item)
		var result = await wordnet.lookupAsync(lemmatizedWord)
		var synonymList = new Set()
		synonymList.add(lemmatizedWord)

		for(var i = result.length - 1; i >= 0; i--){
			for (var m = result[i].synonyms.length - 1; m >= 0; m--) {
				if (result[i].synonyms[m]) {
					synonymList.add(result[i].synonyms[m])
				}
			}
		}
		map.set(lemmatizedWord, Array.from(synonymList))

		result = await wordnet.lookupAsync(item)
		var nextSynonymList = new Set()
		nextSynonymList.add(item)

		for(var i = result.length - 1; i >= 0; i--){
			for (var m = result[i].synonyms.length - 1; m >= 0; m--) {
				if (result[i].synonyms[m] != item) {
					nextSynonymList.add(result[i].synonyms[m])
				}
			}
		}
		map.set(item, Array.from(nextSynonymList))
	}
 	var densityCoefficient = 0
	for (var i = textArray.length - 1; i >= 0; i--) {
		var originalSentenceArr = tokenizeArr(textArray[i])

		var currSentence = tokenize(keyword(textArray[i]))
		if(currSentence.size < 1){
			continue
		}

		var sentenceVector = []
		
		var questionToSentenceMap = []

		var matchList = new Set(
			[...question].filter(x => currSentence.has(x)));

		var qlessc = new Set(
			[...question].filter(x => !currSentence.has(x)));

		var clessq = new Set(
			[...currSentence].filter(x => !question.has(x)));

		matchList.forEach((item) => {
			questionToSentenceMap[item] = item
		})
		qlessc = lemmatizeSet(qlessc)
		clessq = lemmatizeSet(clessq)
		var sharedSize = matchList.size
		
		noMatchList = []
		qlessc.forEach((item) => {
			var synonymList = map.get(item)
			var included = false
			for (var m = synonymList.length - 1; m >= 0; m--) {
				var hasThisSynonym = true
				var synonym = synonymList[m]
				var splitSynonym = synonym.split("_")
				for (var j = splitSynonym.length - 1; j >= 0; j--) {
					if (!clessq.has(splitSynonym[j])) {
						hasThisSynonym = false;
						break;
					}
				}
				if(hasThisSynonym){
					questionToSentenceMap[item] = synonym
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
				noMatchList.push(new Set(synonymList))
			}
		});

		var numTermsMatching = 0
		if(clessq.size > 0) {
 			for (var m = noMatchList.length - 1; m >= 0; m--) {
				bestMatchForEachTerm = 0
				var currArray = Array.from(noMatchList[m])
				for (var j = currArray.length - 1; j >= 0; j--) {
					clessq.forEach((item) => {
						if (item.includes(currArray[j]) && spell.check(item).length > 0) {
							bestMatchForEachTerm = 1;
							questionToSentenceMap[currArray[0]] = item

						}
					})
					if (bestMatchForEachTerm == 1) {
						break;
					}
				}
				numTermsMatching += bestMatchForEachTerm;
			}
		} 
		for(var wordIndex = 0; wordIndex < questionArray.length; wordIndex++){
			if(questionToSentenceMap[questionArray[wordIndex]] != undefined){
				sentenceVector.push(originalSentenceArr.indexOf(questionToSentenceMap[questionArray[wordIndex]]))
			} else {
				questionVector[wordIndex] = -1
			}
		}

		questionVector = questionVector.filter(num => num != -1)
		questionDifferenceArr = arrDifference(questionVector)
		sentenceDifferenceArr = arrDifference(sentenceVector)
		var total = 0;
		for (var elem = questionDifferenceArr.length - 1; elem >= 0; elem--) {
			var raw = Math.abs(questionDifferenceArr[elem] - sentenceDifferenceArr[elem])
			raw = Math.sqrt(raw)
			raw = raw/(Math.pow(questionDifferenceArr[elem],2))
			total += raw
		}
		var orderScore = alteredSigmoid(total)

		console.log("Order: " + orderScore)

		var rating = (sharedSize + numTermsMatching) / (question.size) + densityCoefficient
		if(isNaN(rating)){
			continue;
		}
		
		console.log("Similarity Rating: " + rating)
		var realRating = orderScore * (sharedSize + numTermsMatching) / (question.size) + densityCoefficient
		console.log("Total Rating: " + realRating)
		if(minHeap.size() <= 8) {
			minHeap.insert(rating, textArray[i])
		} else if(rating > minHeap.root().getKey()) {
			minHeap.insert(rating, textArray[i])
			minHeap.extractRoot()
		}
		densityCoefficient = Math.max(densityCoefficient + (rating - .5)/textArray.length, 0)
	}
	var result = []
	for (var i = 7; i >= 0; i--) {
		if(minHeap.size() <= 0){
			break
		}
		result[i] = minHeap.extractRoot().getValue()
	}
	return result

}

getAnswer(question, text)