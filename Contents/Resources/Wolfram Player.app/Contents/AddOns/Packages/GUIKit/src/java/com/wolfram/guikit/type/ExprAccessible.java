/*
 * @(#)ExprAccessible.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.type;

import com.wolfram.jlink.Expr;

/**
 * ExprAccessible is an interface that objects can implement to expose
 * their content as JLink Expr instances and then also to allow
 * parts of their Expr content be set and get just as in Mathematica
 *
 * Implementations should try and support negative indexes like Mathematica
 * requesting elements relative to last element.
 */
public interface ExprAccessible {

	public Expr getExpr();
	public void setExpr(Expr e);

	public Expr getPart(int i);
	public Expr getPart(int[] ia);
	
	public void setPart(int i, Expr e);
	public void setPart(int[] ia, Expr e);
	
  }
