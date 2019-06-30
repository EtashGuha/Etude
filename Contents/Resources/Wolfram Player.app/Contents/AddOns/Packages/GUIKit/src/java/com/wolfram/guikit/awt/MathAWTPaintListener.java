/*
 * @(#)MathAWTPaintListener.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.awt;

import com.wolfram.jlink.MathListener;
import com.wolfram.jlink.KernelLink;

/**
 * MathAWTPaintListener lets you trigger a call into Mathematica on the occurrence of a particular event.
 * Like all the MathXXXListener classes, it is intended to be used primarily from Mathematica, although it
 * can be used from Java code as well.
 * <p>
 * In response to an AWTPaintEvent, objects of this class send to Mathematica:
 * <pre>
 *     userCode[thePaintEvent, thePaintEvent.getGraphics()]</pre>
 * <p>
 * userFunc is specified as a string, either a function name or an expression
 * (like a pure function "foo[##]&"), via the setHandler() method.
 * <p>
 * Two useful articles on what can be done to customize painting of components:
 * http://java.sun.com/products/jfc/tsc/articles/swing2d/index.html
 * http://java.sun.com/products/jfc/tsc/articles/painting/index.html
 *
 */
public class MathAWTPaintListener extends MathListener implements AWTPaintListener {

	/**
	 * The constructor that is called from Mathematica.
	 */
	public MathAWTPaintListener() {
		super();
	}

	/**
	 * You must use this constructor when using this class in a Java program,
	 * because you need to specify the KernelLink that will be used.
	 *
	 * @param ml The link to which computations will be sent when MouseEvents arrive.
	 */
	public MathAWTPaintListener(KernelLink ml) {
		super(ml);
	}

	/**
	 * This form of the constructor lets you skip having
	 * to make a series of setHandler() calls. Use this constructor from Mathematica code only.
	 *
	 * @param handlers An array of {meth, func} pairs associating methods in the MouseListener
	 * interface with Mathematica functions.
	 */
	public MathAWTPaintListener(String[][] handlers) {
		super(handlers);
	}


	////////////////////////////////////  Event handler methods  /////////////////////////////////////////

	public void willPaint(AWTPaintEvent e) {
		callVoidMathHandler("willPaint", prepareArgs(e));
	}

	public void didPaint(AWTPaintEvent e) {
		callVoidMathHandler("didPaint", prepareArgs(e));
	}

	public void willUpdate(AWTPaintEvent e) {
		callVoidMathHandler("willUpdate", prepareArgs(e));
	}

	public void didUpdate(AWTPaintEvent e) {
		callVoidMathHandler("didUpdate", prepareArgs(e));
	}

	private Object[] prepareArgs(AWTPaintEvent e) {
		return new Object[]{e, e.getGraphics()};
	}

}
