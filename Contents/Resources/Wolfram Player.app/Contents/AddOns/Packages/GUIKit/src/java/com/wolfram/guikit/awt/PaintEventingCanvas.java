/*
 * @(#)PaintEventingCanvas.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.awt;

import java.awt.Canvas;
import java.awt.Graphics;
import java.awt.GraphicsConfiguration;

/**
 * PaintEventingCanvas is a simple subclass of Canvas just to
 * hook in events when painting
 */
public class PaintEventingCanvas extends Canvas {

  private static final long serialVersionUID = -1207997970456708948L;
    
  protected AWTPaintEventHandler paintEventHandler = new AWTPaintEventHandler();

  /* Support all constructors of the parent class */

  public PaintEventingCanvas() {
    super();
    }

  public PaintEventingCanvas(GraphicsConfiguration config) {
    super(config);
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
