/*
 * @(#)ExprAccessibleColor.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.awt;

import java.awt.Color;
import java.awt.color.ColorSpace;

import com.wolfram.bsf.util.type.ExprTypeConvertor;
import com.wolfram.guikit.type.ExprAccessible;

import com.wolfram.jlink.Expr;

/**
 * ExprAccessibleColor extends Color
 */
public class ExprAccessibleColor extends Color implements ExprAccessible {
  
    private static final long serialVersionUID = -1887987375486788948L;
  
	private static final ExprTypeConvertor convertor = new ExprTypeConvertor();
	
	public ExprAccessibleColor(float r, float g, float b) {
		super(r,g,b);
		}
	public ExprAccessibleColor(float r, float g, float b, float a) {
		super(r,g,b,a);
		}
	public ExprAccessibleColor(int rgb) {
		super(rgb);
		}
	public ExprAccessibleColor(int rgba, boolean hasalpha) {
		super(rgba, hasalpha);
		}
	public ExprAccessibleColor(int r, int g, int b) {
		super(r,g,b);
		}
	public ExprAccessibleColor(int r, int g, int b, int a) {
		super(r,g,b,a);
		}
	public ExprAccessibleColor(ColorSpace cspace, float components[], float alpha) {
		super(cspace, components, alpha);
		}
		   	    	    	    	    	
	public Expr getExpr() {
	  Object result = convertor.convert(Color.class, Expr.class, this);
	  return (result instanceof Expr) ? (Expr)result : null;
		}
		
	public void setExpr(Expr e) {
		throw new UnsupportedOperationException("Color is immutable and cannot set new values");
		}

	public Expr getPart(int i) {
		Expr result = getExpr();
		if (result != null) return result.part(i);
		return null;
		}
	
	public Expr getPart(int[] ia) {
		Expr result = getExpr();
		if (result != null) return result.part(ia);
		return null;
		}
	
	public void setPart(int i, Expr e) {
		throw new UnsupportedOperationException("Color is immutable and cannot set new values");
		}
		
	public void setPart(int[] ia, Expr e) {
		throw new UnsupportedOperationException("Color is immutable and cannot set new values");
		}
	
  }
