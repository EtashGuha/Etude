/*
 * @(#)SwingPaintListener.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing;

import java.util.EventListener;

/**
 * SwingPaintListener is the listener interface for receiving SwingPaintEvents.
 */
public interface SwingPaintListener extends EventListener {

  /**
   * Called before paintComponent is called on a component.
   * <p>
   * You can request the active graphics context from the paint event and contribute
   * to the painting of the component before it does its own painting.
   *
   * @param e the SwingPaintEvent
   */
  public void willPaintComponent(SwingPaintEvent e);

  /**
   * Called after paintComponent is called on a component.
   * <p>
   * You can request the active graphics context from the paint event and contribute
   * to the painting of the component after it does its own painting.
   *
   * @param e the SwingPaintEvent
   */
  public void didPaintComponent(SwingPaintEvent e);

  /**
   * Called before paintBorder is called on a component.
   * <p>
   * You can request the active graphics context from the paint event and contribute
   * to the painting of the component before it paints its border.
   *
   * @param e the SwingPaintEvent
   */
  public void willPaintBorder(SwingPaintEvent e);

  /**
   * Called after paintBorder is called on a component.
   * <p>
   * You can request the active graphics context from the paint event and contribute
   * to the painting of the component after it paints its border.
   *
   * @param e the SwingPaintEvent
   */
  public void didPaintBorder(SwingPaintEvent e);

  /**
   * Called before paintChildren is called on a component.
   * <p>
   * You can request the active graphics context from the paint event and contribute
   * to the painting of the component before it paints its children components.
   *
   * @param e the SwingPaintEvent
   */
  public void willPaintChildren(SwingPaintEvent e);

  /**
   * Called after paintChildren is called on a component.
   * <p>
   * You can request the active graphics context from the paint event and contribute
   * to the painting of the component after it paints its children components.
   *
   * @param e the SwingPaintEvent
   */
  public void didPaintChildren(SwingPaintEvent e);

  }
