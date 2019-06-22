package com.wolfram.stanfordnlp;

import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.ling.Word;
import java.util.Queue;

import java.io.StringReader;

import java.lang.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Collections;
import java.util.stream.*;

/**
 * StanfordNLPPOSWrapper
 *
 **/

public class StanfordNLPPOSWrapper
{
	private List<Tree> treeList;
	private List<int[][]> childIndexes;
	private List<String[]> labels;
	private List<double[]> scores;
	
	public StanfordNLPPOSWrapper(LexicalizedParser lexicalParser, String[][] stringsIn){
		
		List<List<Word>> sentences = new ArrayList(Collections.<List<Word>>emptyList());
		
		for(int i=0; i<stringsIn.length; i++){
			String[] tokens = stringsIn[i];
			List<Word> words = new ArrayList<Word>();
			
			for(int j=0; j<tokens.length; j++){
				words.add(new Word(tokens[j]));
			}
		
			sentences.add(words);
		};
		
		treeList = lexicalParser.parseMultiple(sentences);
		
		childIndexes = new ArrayList<int[][]>();
		labels = new ArrayList<String[]>();
		scores = new ArrayList<double[]>();
		
		for(int t=0; t<treeList.size(); t++) {
			Tree tree = treeList.get(t);
			Queue<Tree> queue = new java.util.LinkedList<Tree>();
			queue.offer(tree);
			int sentInQueue = 1;
			
			List<int[]> treeChildIndexes = new ArrayList<int[]>();
			List<String> treeLabels = new ArrayList<String>();
			List<Double> treeScores = new ArrayList<Double>();
			
			while(!queue.isEmpty()){
				Tree node = queue.poll();
				int childcount = node.numChildren();
				int[] childs = new int[childcount];
				if(childcount>0){
					for (int i=0; i<childcount; i++){
						queue.offer(node.getChild(i));
						sentInQueue++;
						childs[i] = sentInQueue;
					}
				}
				treeChildIndexes.add(childs);
				treeLabels.add(node.label().value());
				treeScores.add(node.score());
			}
			
			childIndexes.add(treeChildIndexes.toArray(new int[treeChildIndexes.size()][]));
			labels.add(treeLabels.toArray(new String[treeLabels.size()]));
			scores.add(treeScores.stream().mapToDouble(Double::doubleValue).toArray());
		}
	}
	
	public int[][][] getChildIndexes(){
		return this.childIndexes.toArray(new int[0][][]);
	}
	
	public String[][] getLabels(){
		return this.labels.toArray(new String[0][]);
	}
	
	public double[][] getScores(){
		return this.scores.toArray(new double[0][]);
	}
	
	
}