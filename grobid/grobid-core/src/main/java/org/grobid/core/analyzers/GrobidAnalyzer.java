package org.grobid.core.analyzers;

import org.grobid.core.lang.Language;
import org.grobid.core.layout.LayoutToken;

import org.grobid.core.utilities.UnicodeUtil;
import org.wipo.nlp.textboundaries.ReTokenizer;
import org.wipo.nlp.textboundaries.ReTokenizerFactory;

import java.util.List;
import java.util.ArrayList;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An Analyzer able to dispatch text to be tokenized to the adequate analyzer given a specified language.
 *
 * The language might be preliminary set by the language recognizer or manually if it is already 
 * known by the context of usage of the text. 
 *
 * @author Patrice Lopez
 */
public class GrobidAnalyzer implements Analyzer {
	private static final Logger LOGGER = LoggerFactory.getLogger(GrobidAnalyzer.class);
	
	private static volatile GrobidAnalyzer instance;
	
	private ReTokenizer jaAnalyzer = null;
	private ReTokenizer krAnalyzer = null;
	private ReTokenizer zhAnalyzer = null;
	
    public static GrobidAnalyzer getInstance() {
        if (instance == null) {
            //double check idiom
            // synchronized (instanceController) {
                if (instance == null)
					getNewInstance();
            // }
        }
        return instance;
    }

    /**
     * Creates a new instance.
     */
	private static synchronized void getNewInstance() {
		LOGGER.debug("Get new instance of GrobidAnalyzer");
		instance = new GrobidAnalyzer();
	}

    /**
     * Hidden constructor
     */
    private GrobidAnalyzer() {
	}
	
	public String getName() {
		return "GrobidAnalyzer";
	} 

	/**
	 * Tokenizer entry point
	 */
	public List<String> tokenize(String text) {
		return tokenize(text, null);
	}
	
	public List<String> tokenize(String text, Language lang) {
		List<String> result = new ArrayList<String>();
		if ( (text == null) || (text.length() == 0) ) {
			return result;
		}
		try {
			if ( (lang == null) || (lang.getLang() == null) ) {
				// default Indo-European languages
				result = GrobidDefaultAnalyzer.getInstance().tokenize(text);
			}
			else if (lang.isJapaneses()) {
				// Japanese analyser
				if (jaAnalyzer == null)
					jaAnalyzer = ReTokenizerFactory.create("ja_g");
				result = jaAnalyzer.tokensAsList(text);
			}
			else if (lang.isChinese()) {
				// Chinese analyser
				if (zhAnalyzer == null)
					zhAnalyzer = ReTokenizerFactory.create("zh_g");
				result = zhAnalyzer.tokensAsList(text);
			}
			else if (lang.isKorean()) {
				// Korean analyser
				if (krAnalyzer == null)
					krAnalyzer = ReTokenizerFactory.create("kr_g");
				result = krAnalyzer.tokensAsList(text);
			}
			else if (lang.isArabic()) {
				// Arabic analyser
				result = GrobidDefaultAnalyzer.getInstance().tokenize(text);
				int p = 0;
				for(String token : result) {
					// string being immutable in Java, I think we can't do better that this:
					StringBuilder newToken = new StringBuilder();
					for(int i=0; i<token.length(); i++) {
						newToken.append(ArabicChars.arabicCharacters(token.charAt(i)));
					}
					result.set(p, newToken.toString());
					p++;
				}
			}
			else {
				// default Indo-European languages
				result = GrobidDefaultAnalyzer.getInstance().tokenize(text);
			}
		} catch(Exception e) {
			LOGGER.error("Invalid tokenizer", e);
		}
		return result;
	}
	
	/**
	 * Re-tokenizer entry point to be applied to text already tokenized in the PDF representation
	 */
	public List<String> retokenize(List<String> textTokenized) {
		return retokenize(textTokenized, null);
	}
	
	public List<String> retokenize(List<String> textTokenized, Language lang) {
		List<String> result = null;
		if ( (textTokenized == null) || (textTokenized.size() == 0) ) {
			return new ArrayList<String>();
		}
		if ( (lang == null) || (lang.getLang() == null) ) {
			// default Indo-European languages
			result = GrobidDefaultAnalyzer.getInstance().retokenize(textTokenized);
		}
		else if (lang.isJapaneses()) {
			// Japanese analyser
		}
		else if (lang.isChinese()) {
			// Chinese analyser
		}
		else if (lang.isKorean()) {
			// Korean analyser
		}
		else if (lang.isArabic()) {
			// Arabic analyser
		}
		else {
			// default Indo-European languages
			result = GrobidDefaultAnalyzer.getInstance().retokenize(textTokenized);
		}
		return result;
	}

	public List<LayoutToken> tokenizeWithLayoutToken(String text) {
		return tokenizeWithLayoutToken(text, null);
	}

	public List<LayoutToken> tokenizeWithLayoutToken(String text, Language lang) {
        List<LayoutToken> result = new ArrayList<>();
        text = UnicodeUtil.normaliseText(text);
        List<String> tokens = tokenize(text, lang);
        int pos = 0;
        for (int i = 0; i < tokens.size(); i++) {
            String tok = tokens.get(i);
            LayoutToken layoutToken = new LayoutToken();
            layoutToken.setText(tok);
            layoutToken.setOffset(pos);
            result.add(layoutToken);
            pos += tok.length();
            if (i < tokens.size() - 1 && tokens.get(i + 1).equals("\n")) {
                layoutToken.setNewLineAfter(true);
            }
        }

        return result;
    }
}