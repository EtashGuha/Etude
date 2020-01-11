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

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
var question = "progressive ideology humanism"
var text = "Almost all Americans share some elements of a common political culture. Why, then, is there so much cultural conflict in American politics? For many years, the most explosive political issues have included abortion, gay rights, drug use, school prayer, and pornography. Viewed from a Marxist perspective, politics in the United States is utterly baffling: instead of two economic classes engaged in a bitter struggle over wealth, we have two cultural classes locked in a war over values. As first formulated by sociologist James Davison Hunter, the idea is that there are, broadly defined, two cultural classes in the United States: the orthodox and the progressive. On the orthodox side are people who believe that morality is as important as, or more important than, self-expression and that moral rules derive from the commands of God or the laws of nature—commands and laws that are relatively clear, unchanging, and independent of individual preferences. On the progressive side are people who think that personal freedom is as important as, or more important than, certain traditional moral rules and that those rules must be evaluated in light of the circumstances of modern life—circumstances that are quite complex, changeable, and dependent on individual preferences.31 Most conspicuous among the orthodox are fundamentalist Protestants and evangelical Christians, and so critics who dislike orthodox views often dismiss them as the fanatical expressions of “the Religious Right.” But many people who hold orthodox views are not fanatical or deeply religious or rightwing on most issues: they simply have strong views about drugs, pornography, and sexual morality. Similarly, the progressive side often includes members of liberal Protestant denominations (for example, Episcopalians and Unitarians) and people with no strong religious beliefs, and so their critics often denounce them as immoral, anti-Christian radicals who have embraced the ideology of secular humanism, the belief that moral standards do not require religious justification. But in all likelihood few progressives are immoral or anti-Christian, and most do not regard secular humanism as their defining ideology."
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
	originalQuestionArr = tokenizeArr(question.toLowerCase())
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
		var copyQuestionVector = questionVector
		for(var wordIndex = 0; wordIndex < questionArray.length; wordIndex++){
			if(questionToSentenceMap[questionArray[wordIndex]] != undefined){
				sentenceVector.push(originalSentenceArr.indexOf(questionToSentenceMap[questionArray[wordIndex]]))
			} else {
				copyQuestionVector[wordIndex] = -1
			}
		}

		copyQuestionVector = copyQuestionVector.filter(num => num != -1)
		questionDifferenceArr = arrDifference(copyQuestionVector)
		sentenceDifferenceArr = arrDifference(sentenceVector)

		var total = 0;
		for (var elem = questionDifferenceArr.length - 1; elem >= 0; elem--) {
			var raw = Math.abs(questionDifferenceArr[elem] - sentenceDifferenceArr[elem])
			raw = Math.sqrt(raw)
			raw = raw/(Math.pow(questionDifferenceArr[elem],2))
			total += raw
		}
		var orderScore = alteredSigmoid(total)
		
		var realRating = (orderScore + 1) * (sharedSize + numTermsMatching) / (question.size) + densityCoefficient

		if(isNaN(realRating)){
			continue;
		}

		if(minHeap.size() <= 8) {
			minHeap.insert(realRating, textArray[i])
		} else if(realRating > minHeap.root().getKey()) {
			minHeap.insert(realRating, textArray[i])
			minHeap.extractRoot()
		}
		densityCoefficient = Math.max(densityCoefficient + (realRating - .5)/textArray.length, 0)
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

getAnswer(question, text).then((data)=> {
	console.log(data)
})