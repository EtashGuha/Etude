/*
 * @(#)ParameterListener.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.trek.event;

import java.util.EventListener;

/**
 * ParameterListener is the listener interface for receiving parameter events.
 */
public interface ParameterListener extends EventListener {

  /**
   * Called after a paramter's value has changed
   * <p>
   *
   * @param e the ParameterEvent
   */
  public void didChange(ParameterEvent e);

  }
