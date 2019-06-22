/*
 * @(#)PaintEventingMathCanvas.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.awt;

import java.awt.Graphics;

import com.wolfram.jlink.KernelLink;
import com.wolfram.jlink.MathCanvas;

/**
 * PaintEventingMathCanvas is a simple subclass of MathCanvas just to
 * hook in events when painting
 */
public class PaintEventingMathCanvas extends MathCanvas {

  private static final long serialVersionUID = -1217987915456188148L;
    
  protected AWTPaintEventHandler paintEventHandler = new AWTPaintEventHandler();

  /* Support all constructors of the parent class */

  public PaintEventingMathCanvas() {
    super();
    }

  public PaintEventingMathCanvas(KernelLink ml) {
    super(ml);
    }

  /* Support for events during painting */

  public void paint(Graphics g) {
    paintEventHandler.fireWillPaint(g, this);
    super.paint(g);
    paintEventHandler.fireDidPaint(g, this);
    }

  public void update(Graphics g) {
    paintEventHandler.fireWillUpdate(g, this);
    super.update(g);
    paintEventHandler.fireDidUpdate(g, this);
    }

	/**
	 * Adds the specified AWTPaintListener to receive AWTPaintEvents.
	 * <p>
	 * Use this method to register an AWTPaintListener object to receive
	 * notifications when painting occurs on this component
	 *
	 * @param listener the AWTPaintListener to register
	 * @see #removePaintListener(AWTPaintListener)
	 */
  public void addPaintListener(AWTPaintListener listener) {
    paintEventHandler.addPaintListener(listener);
    }

	/**
	 * Removes the specified AWTPaintListener object so that it no longer receives
	 * AWTPaintEvents.
	 *
	 * @param listener the AWTPaintListener to register
	 * @see #addPaintListener(AWTPaintListener)
	 */
  public void removePaintListener(AWTPaintListener listener) {
    paintEventHandler.removePaintListener(listener);
    }

  }
