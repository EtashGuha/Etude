/*
 * @(#)AWTPaintListener.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.awt;

import java.util.EventListener;

/**
 * AWTPaintListener is the listener interface for receiving AWTPaintEvents.
 */
public interface AWTPaintListener extends EventListener {

  /**
   * Called before paint is called on a component.
   * <p>
   * You can request the active graphics context from the paint event and contribute
   * to the painting of the component before it does its own painting.
   *
   * @param e the AWTPaintEvent
   */
  public void willPaint(AWTPaintEvent e);

  /**
   * Called after paint is called on a component.
   * <p>
   * You can request the active graphics context from the paint event and contribute
   * to the painting of the component after it does its own painting.
   *
   * @param e the AWTPaintEvent
   */
  public void didPaint(AWTPaintEvent e);

  /**
   * Called before update is called on a component.
   * <p>
   * You can request the active graphics context from the paint event and contribute
   * to the painting of the component before update is called.
   *
   * @param e the AWTPaintEvent
   */
  public void willUpdate(AWTPaintEvent e);

  /**
   * Called after update is called on a component.
   * <p>
   * You can request the active graphics context from the paint event and contribute
   * to the painting of the component after update is called.
   *
   * @param e the AWTPaintEvent
   */
  public void didUpdate(AWTPaintEvent e);

  }
