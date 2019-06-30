/*
 * @(#)PaintEventingJPanel.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing;

import java.awt.LayoutManager;
import java.awt.Graphics;

import javax.swing.JPanel;

/**
 * PaintEventingJPanel is a simple subclass of JPanel just to
 * hook in events when painting
 */
public class PaintEventingJPanel extends JPanel {

  private static final long serialVersionUID = -1287987975457788748L;
    
  protected SwingPaintEventHandler paintEventHandler = new SwingPaintEventHandler();

  /* Support all constructors of the parent class */

  public PaintEventingJPanel(LayoutManager layout, boolean isDoubleBuffered) {
    super(layout, isDoubleBuffered);
    }

  public PaintEventingJPanel(LayoutManager layout) {
    super(layout);
    }

  public PaintEventingJPanel(boolean isDoubleBuffered) {
    super(isDoubleBuffered);
    }

  public PaintEventingJPanel() {
    super();
    }

  /* Support for events during painting */

  public void paintComponent(Graphics g) {
    paintEventHandler.fireWillPaintComponent(g, this);
    super.paintComponent(g);
    paintEventHandler.fireDidPaintComponent(g, this);
    }

  public void paintBorder(Graphics g) {
    paintEventHandler.fireWillPaintBorder(g, this);
    super.paintBorder(g);
    paintEventHandler.fireDidPaintBorder(g, this);
    }

  public void paintChildren(Graphics g) {
    paintEventHandler.fireWillPaintChildren(g, this);
    super.paintChildren(g);
    paintEventHandler.fireDidPaintChildren(g, this);
    }

	/**
	 * Adds the specified SwingPaintListener to receive SwingPaintEvents.
	 * <p>
	 * Use this method to register a SwingPaintListener object to receive
	 * notifications when painting occurs on this component
	 *
	 * @param listener the SwingPaintListener to register
	 * @see #removePaintListener(SwingPaintListener)
	 */
  public void addPaintListener(SwingPaintListener listener) {
    paintEventHandler.addPaintListener(listener);
    }

	/**
	 * Removes the specified SwingPaintListener object so that it no longer receives
	 * SwingPaintEvents.
	 *
	 * @param listener the SwingPaintListener to register
	 * @see #addPaintListener(SwingPaintListener)
	 */
  public void removePaintListener(SwingPaintListener listener) {
    paintEventHandler.removePaintListener(listener);
    }

  }
