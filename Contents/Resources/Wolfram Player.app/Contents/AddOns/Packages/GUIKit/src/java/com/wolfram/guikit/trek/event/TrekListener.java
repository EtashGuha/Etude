/*
 * @(#)TrekListener.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.trek.event;

import java.util.EventListener;

/**
 * TrekListener is the listener interface for treks
 */
public interface TrekListener extends EventListener {

  /**
   * Called after a paramter's value has changed
   * <p>
   *
   * @param e the TrekEvent
   */
  public void trekOriginDidChange(TrekEvent e);
	public void trekIndependentRangeDidChange(TrekEvent e);
	
  }
