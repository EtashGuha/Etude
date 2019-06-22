package com.wolfram.stanfordnlp;

import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.ling.CoreLabel;
import java.io.StringReader;

import java.lang.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Collections;

/**
 * StanfordNLPWrapper
 *
 **/
public class StanfordNLPWrapper
{
	//La collection de labels en cours
	private List<List<CoreLabel>> tokens = new ArrayList(Collections.<List<CoreLabel>>emptyList());
	
	public StanfordNLPWrapper(CoreLabelTokenFactory coreTokenFactory, String[] stringsIn, String options){
		
		for(int i=0; i<stringsIn.length; i++){
			
			PTBTokenizer tokenizer = new PTBTokenizer(new StringReader(stringsIn[i]), coreTokenFactory, options);
			tokens.add(tokenizer.tokenize());
		}
		
	}
	
	public int getTokensLength(){
			return this.tokens.size();
	}
	
	public Integer[][][] getPositions() {
		Integer[][][] positionsAr = new Integer[this.tokens.size()][][];
		
		for (int i = 0; i < this.tokens.size(); i++){
			List<CoreLabel> strtokens = this.tokens.get(i);
			ArrayList<Integer[]> strPositions = new ArrayList(Collections.<Integer[]>emptyList());
			
			for (int j = 0; j < strtokens.size(); j++){
				CoreLabel coreLabel = strtokens.get(j);
				strPositions.add(new Integer[] {(Integer) coreLabel.beginPosition(), (Integer) coreLabel.endPosition()}) ;
			}
			
			positionsAr[i] = strPositions.toArray(new Integer[][] {});
		}
		
		return positionsAr;
	}
	
	public String[][][] getNormalizedRawBeforeAfterTokens() {
		String[][][] tokensAr = new String[this.tokens.size()][][];
		
		for (int i = 0; i < this.tokens.size(); i++){
			List<CoreLabel> strtokens = this.tokens.get(i);
			ArrayList<String[]> strTokens = new ArrayList(Collections.<String[]>emptyList());
			
			for (int j = 0; j < strtokens.size(); j++){
				CoreLabel coreLabel = strtokens.get(j);
				strTokens.add(new String[] { coreLabel.value(), coreLabel.originalText(), coreLabel.before(), coreLabel.after() }) ;
			}
			
			tokensAr[i] = strTokens.toArray(new String[][] {});
		}
		
		return tokensAr;
	}
	
    
}