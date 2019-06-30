/*
 * @(#)ExprGraphTypeConvertor.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.graph;

import com.wolfram.bsf.util.type.MathematicaTypeConvertor;
import com.wolfram.jlink.Expr;
import com.wolfram.jlink.LoopbackLink;
import com.wolfram.jlink.MathLink;
import com.wolfram.jlink.MathLinkException;
import com.wolfram.jlink.MathLinkFactory;

import diva.graph.GraphModel;
import diva.graph.basic.BasicGraphModel;

import java.math.BigDecimal;
import java.math.BigInteger;

/**
 * ExprGraphTypeConvertor
 */
public class ExprGraphTypeConvertor implements MathematicaTypeConvertor {

	public Object convert(Class from, Class to, Object obj) {

		if (to == Expr.class) {
			if (from == Expr.class) {
				return obj;
				}
				
			else if (GraphModel.class.isAssignableFrom(from)) {
				
				Expr e = null;
				GraphModel t = (GraphModel)obj;
				LoopbackLink loop = null;
				try {
					loop = MathLinkFactory.createLoopbackLink();
					putGraphNode(loop, t, t.getRoot());
					e = loop.getExpr();
					}
				catch (MathLinkException me) {}
				finally {
					if (loop != null)
						loop.close();
					}
				return e;
				
				}
		
	    else {
	      return null;
	      }
    	}
		else if (from == Expr.class) {
			try {
			if (to == Expr.class) {
				return obj;
				}

			else if (GraphModel.class.isAssignableFrom(to)) {
				GraphModel m = new BasicGraphModel();
				return m;
				}

			else {
				return null;
				}
			}
		 catch(Exception e) {return null;}
			}
		else
			return null;
	}

  public Object convertExprAsContent(Expr e) {
  	if (e.bigDecimalQ()) {
  		return convert(Expr.class, BigDecimal.class, e);
  		}
		else if (e.bigIntegerQ()) {
			return convert(Expr.class, BigInteger.class, e);
			}
		else if (e.stringQ()) {
			return convert(Expr.class, String.class, e);
			}
		else if (e.equals(Expr.SYM_TRUE) || e.equals(Expr.SYM_FALSE)) {
			return convert(Expr.class, Boolean.class, e);
			}
		else if (e.realQ()) {
			return convert(Expr.class, Double.class, e);
			}
		else if (e.integerQ()) {
			return convert(Expr.class, Integer.class, e);
			}
  	return e;
  	}
  	
  // Not implemented or used anywhere
  public String getCodeGenString () {
    return "";
    }
 
  	
	protected void putGraphNodeContent(MathLink link, Object node) throws MathLinkException {
		//	TODO currently having to use put() forces non JLink converted
		// Mathematica types to Strings instead of being able to do putReference()
		// on a KernelLink to preserve some elements as JavaObjects if more complex
		//if (node instanceof DefaultMutableTreeNode) {
		//	link.put( ((DefaultMutableTreeNode)node).getUserObject());
		//	}
		//else link.put( node);
		}
		
		
  protected void putGraphNode(MathLink link, GraphModel model, Object node) throws MathLinkException {
  	if (node == null) {
  		link.putSymbol("Null");
  		}
  	//else if (node instanceof TreeNode && !((TreeNode)node).getAllowsChildren()) {
  	//	putGraphNodeContent(link, node);
  	//	}
  	else {
			link.putFunction("List", 2);
			putGraphNodeContent(link, node);
			//int count = model.getChildCount(node);
			int count = 0;
			link.putFunction("List", count);
			for (int i = 0; i < count; ++i) {
				//putGraphNode(link, model, model.getChild(node, i));
				}	
  		}
  	}
  
}

