const lemmatizer = require("lemmatizer")
var thesaurus = require("thesaurus");
var stringSimilarity = require('string-similarity');
var HashMap = require('hashmap');
var spellChecker = require('spellchecker')
const {
	MinHeap,
	MaxHeap
} = require('@datastructures-js/heap');
const minHeap = new MinHeap();

var map = new HashMap();

var text = "English    Professor Rittenhouse   February     Etash Guha   NonTraditional Narrative Elements in   Moby Dick   In   Moby Dick   Herman Melville often deviates from the traditional narrative format. For   example one might turn to any given page within the book and believe that they are reading an   encyclopedia or a Shakespearean play. In addition the subject matter of the chapters isnt always   based on the main plot. Sometimes Melville interjects certain passages that have very little to do   with the main plotline catching the infamous white whale that is Moby Dick. To this point some   critics believe that Melvilles use of these different forms and subjects is a distraction from the   main narrative. They argue that the book would be better off without these seemingly random   additions of nonplot or nonnarrative material. However these additions in fact do provide   meaningful information to the reader in ways that enhance rather than degrade the central   narrative.  In fact while some claim that Herman Melvilles use of nonplot related material   takes away from the true story it provides a more thorough understanding of the story and a   stronger connection with the characters.   httpswww.youtube.comwatch?vWabTLnNE  . For   example if one were to click on this link and close their eyes theyd feel like they were on the   boat with Ishmael. A similar effect is achieved by the aspects of drama cetology and whaling   within the book.   Throughout   Moby Dick   drama     is an effective tool for characterization within the story   giving the reader valuable insight into the characters. One of the most common aspects of drama  that Melville employs is that of the soliloquy specifically that of the Shakespearean soliloquy   such as in famous plays like   Macbeth  . One notable example of Melvilles use of the soliloquy is   that in Chapter  Sunset where Ahab displays a different side of his character that the reader   had not previously seen before. Before his soliloquy Ahab was portrayed as apathetic a form of   authority since he was the Khan of the plank and a king of the sea... great lord of Leviathans   Melville . Since the narrative form is limited to Ishmaels point of view the narrative form   might have not been able to efficiently display a side of Ahab that is anything but his outward   appearance what he wants the crew to see him as. However through the soliloquy we are able   to see Ahabs inner thought process as he gives a great speech of resolution one in which he   clamps down upon his weakness of self Vogel . Indeed the use of the dramatic form   gives the reader a deeper insight into Ahabs character. The reader is able to better understand   how the whale is his motive and makes him feel and what his emotions and thoughts towards his   crewmates are.     In addition Pip is a character that Melville uses for dramatization. Pip is a character who   leaps out of the boat when the whale hits the boat and thus is a portrayal of fear. When he gives   his speech about the big white God he gives  the curtain speech of the actVogel . He   uses the characteristic of a curtain speech to define the catalyst of conflict of   characterfearVogel  that fear being that of losing ones life to the sea that is so   commonly held by each member of the crew. This curtain speech establishes the totality of this   fear and gives the reader insight into the main fear of the characters. In addition Chapter  is   particularly similar to that of a play it includes the roles of who is saying each phrase and where   they are from. In this way it establishes the international background of the crew highlighting  each crew member. It gives the reader information about the crew that might have been difficult   or awkward to portray in the form of the traditional narrative. Certainly it would have been more   difficult to incorporate dialogue from so many of the characters in a narrative form instead of a   play form. This information about the characters is unique to the perspective of drama. If   instead Melville was to stay in the narrative form and within Ishmaels perspective these   insights would have been lost. The form of drama is better suited to display information about   the characters and different viewpoints than a narrative is.  In this manner the dramatic elements   are an effective method of evoking emotion and characterization.     While Melville would stray from the narrative form with his use of dramatic elements he   also strayed away from the main storyline in order to discuss the importance and intricacies of   cetology. One of the most significant chapters to discuss cetology was the aptly named chapter   Cetology. Many critics and readers of   Moby Dick   believed these chapters about cetology to be   an incongruous blend of formal exposition and traditional narration finding it difficult to   accept this unique use of nonfiction Ward . These chapters give the reader the effect of a   long voyage despite the obvious fact that very little happens when on a whaling journey   Ward . This gives the reader a more realistic sense of the story lending verisimilitude to   the story Hilbert . As the reader is educated about cetology the monotony and mindset of   the whalers becomes more and more apparent. As the story mimics nonfiction the reader begins   to associate what is happening in the book with nonfiction making the narrative seem more   realistic. In addition to giving a more realistic sense of the dreadfulness and monotony of the   journey it also lends to Ishmaels characterization. It displays his values in education and his   background such as displaying his knowledge of whaling or his childlike tendency to daydream  Hilbert . While most narratives would forgo this information as it is unrelated to the plot   this addition leads the reader to better understand who Ishmael is. On a simpler level the   expository information provides crucial context to the reader about the story. Without this   information it is likely the reader would be left confused by the whaling jargon Ward . One   such example where the cetology gives crucial context to the story is when Tashtego falls into   the head of the whale the chapter before describes the anatomical and dimensional structure of   the whales head which allows the reader to visualize Tashtegos fall. In this way the   information from the nonfiction and the narrative work together to convey a clearer message to   the reader Ward .   The cetology in the book also gives the reader a familiarity with the whale specifically   the sperm whale that wouldnt have come without such descriptions. For example by the end of   Chapter  the reader already has a visual image of the whale one of the largest inhabitants of   the world and that of an AnvilHeaded Whale Melville . The analysis of the anatomical   structures social tendencies eating habits and more makes the whale more comprehensible and   understandable from the readers point of view Ward . This familiarity of the whale   contrasts that of the mythical and fantastic nature associated with the whale providing a larger   more developed view of it. Like that of the monotony of the journey Melvilles description of   the whale drags it into the readers world and makes the narrative seem that much more real.   This connection and perspective of the narrative are unique to the combination of nonfiction and   fiction. This analysis of the whale exists on social physical biological and other different types   of levels developing the whale as a frame of reference for the man Ward . In fact there   are several similarities between Ishmaels situation and that of a whale that this cetology chapter  points out. For example as evil is present in the world... the sharks snap about the whale   Ward . This meaningful analogy between the whale and humankind is produced by this   cetological chapter adding to the narrative by providing a different perspective than would have   normally been presented. While the whale might have been reserved to the archetypal antagonist   in the story the nonfiction creates this analogy and adds a different perspective of the whale that   enriches the narrative. Thus the cetological elements of   Moby Dick   are effective tools that   Melville uses to enhance the main narrative.   In addition to the material about whales giving the reader information the material about   the craft of whaling gives the reader a perspective that would not have been possible otherwise.   One such example several versions of   Moby Dick   provide illustrations of several aspects of   whaling itself. For example this image details what the Pequod the ship that Ishmael is on look   likes. While not relevant and necessary to the typical structure of a narrative this image provides   great detail and   context to Ishmaels   situation. It provides   the reader with a   view of what his life   would be like what   it visually looked   like. This is quite   fundamentally impossible with the typical structure of a narrative. Thus this atypical   methodology of providing information about whaling enhances the narrative well. While  Melville did not himself add the image the editor added it as heshe felt it enhanced the narrative   for the detail and context it provides to the readers and as it improved the narrative overall. His   desire to use this aspect of nonfiction to provide the reader with a visual image of Ishmaels   world displays nonfictions value in improving the narrative. Also the chapters in   Moby Dick   from around Chapter  to  detail in great length what the life of a whaler consists of giving   the reader a far more involved look into the lives of Ishmael and company. For example the   chapter The Battering Ram prepares the reader to accept the power and apparent malignity of   Moby Dick which will seal the fate of Pequod an unfortunate but possible part of whaling   Ward . This physical description of the risks that whalers undergo gives the reader the sense   of fear and apprehension that many whalers live with constantly. This allows the reader to not   only acknowledge the emotions of the characters but also empathize as they can translate the   characters fears into their own life since it is nonfiction and more accurately portrays life. In this   way the nonfiction chapters about whaling in   Moby Dick   translate the emotions and ideas of   Ishmaels world into the world of the reader giving them a deeper and more intimate connection   with the characters themselves.   In addition the whaling chapters also give the reader a more personal understanding of   Melvilles personal views. Their venture into the real world allows for Melville to incorporate   his philosophies and convey them to the reader. For example the chapter of the Mast Head   displays the monotony and troubles that manning the mast and looking for whales can have.   While Melville displays the issues that Ishmael has with being on the masthead he displays his   views and ridicules the subjectivism of transcendentalism as it causes him to mistake the   appearance of things for the truth of things Ward . He does this by illustrating how  Ishmaels dreams while on top of the masthead distract him from the realistic world. In this   manner Melville cleverly integrates his personal views with the story. This deeper familiarity   with the author contrasts the alienated position that he would have held if he had stuck to the   traditional narrative style. In this manner the chapters about whaling give the reader a view that   supplements the main narrative elements.   These nonnarrative elements of drama cetology and whaling give the reader a valuable   understanding and connection with the story that acts to supplement the more traditional   narrative elements. In this way Melville can represent both the materialist view through the   familiar form of a veritable record and the idealist or visionary view through more figurative   language PostLauria . This combination of forms provides the readers with a connection   with the characters and author. It also urges the modern reader to consider this matter of mixed   form ... in every light PostLauria . Thus these aspects of the book are valuable   inclusions to do without them would take from the meaning and journey of the book itself.                                Works Cited   Hilbert Betsy. The Truth of the Thing Nonfiction in MobyDick.   College English   vol.  no.     pp. .   JSTOR     www.jstor.orgstable  .   PostLauria Sheila. Philosophy in Whales... Poetry in Blubber Mixed Form in MobyDick.   NineteenthCentury Literature   vol.  no.   pp. .   JSTOR     www.jstor.orgstable  .   Vogel Dan. The Dramatic Chapters in Moby Dick.   NineteenthCentury Fiction   vol.  no.     pp. .   JSTOR     www.jstor.orgstable  .   Ward J. A. The Function of the Cetological Chapters in MobyDick.   American Literature   vol.    no.   pp. .   JSTOR   www.jstor.orgstable.  "
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
var question = "What is the point of view of thenarration?"

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

async function cleanText(inSet) {
	var outSet = new Set()
	inSet.forEach((item) => {
		if(spellChecker.isMisspelled(item)){
			require('google-autosuggest')(item).then(resp => {
  				console.log(resp)
  				return resp
			})
		}
	})
}
async function getAnswer(question, text){
	var textArray = text.toLowerCase().match(/[^\.!\?]+[\.!\?]+/g)
	question = tokenize(question)

	question.forEach((item) => {
		map.set(lemmatizer.lemmatizer(item), thesaurus.find(lemmatizer.lemmatizer(item)))
		map.set(item, thesaurus.find(item))
	})


	for (var i = textArray.length - 1; i >= 0; i--) {

		var currSentence = tokenize(textArray[i])
		var matchList = new Set(
			[...question].filter(x => currSentence.has(x)));

		var qlessc = new Set(
			[...question].filter(x => !currSentence.has(x)));

		var clessq = new Set(
			[...currSentence].filter(x => !question.has(x)));

		qlessc = lemmatizeSet(qlessc)
		clessq = lemmatizeSet(clessq)
		await cleanText(clessq)
		// console.log(clessq)
		var sharedSize = matchList.size
		var noMatchList = []
		qlessc.forEach((item) => {
			// console.log(item)
			var synonymList = map.get(item)
			var included = false
			for (var i = synonymList.length - 1; i >= 0; i--) {
				var synonym = synonymList[i]
				var splitSynonym = synonym.split(" ")
				for (var j = splitSynonym.length - 1; j >= 0; j--) {
					if (clessq.has(splitSynonym[j])) {
						included = true;
						clessq.delete(splitSynonym[j])
						break;
					}
				}
			}
			if (included) {
				sharedSize += 1;
			} else {
				noMatchList = noMatchList.concat(synonymList)
				noMatchList.push(item)
			}
		});
		var sumOfMatches = 0;
		if (noMatchList.length != 0) {
			clessq.forEach((item) => {
				var bestMatch = stringSimilarity.findBestMatch(item, noMatchList).bestMatch
				sumOfMatches += bestMatch.rating
			})
		}

		var rating = sharedSize + (sumOfMatches/clessq.size) / (question.size)

		if(isNaN(rating)){
			console.log("NOPEE")
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