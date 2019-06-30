/*
 * @(#)AWTPaintEventHandler.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.trek.event;

import javax.swing.event.EventListenerList;

import com.wolfram.guikit.trek.Parameter;

/**
 * AWTPaintEventHandler is a utility class that wraps all the code needed to process
 * and manage AWT paint events and the listeners.
 * <p>
 * Use of this class simplifies the code needed in subclasses of AWT components
 * that want to implement AWTPaintEvents
 */
public class ParameterEventHandler  {

  protected EventListenerList listeners = null;

	/**
	 * Adds the specified ParameterListener to receive ParameterEvents.
	 * <p>
	 * Use this method to register a ParameterListener object to receive
	 * notifications when parameter changes occur
	 *
	 * @param l the ParameterListener to register
	 * @see #removeParameterListener(ParameterListener)
	 */
  public void addParameterListener(ParameterListener l) {
    if (listeners == null) listeners = new EventListenerList();
    if ( l != null ) {
      listeners.add( ParameterListener.class, l );
      }
    }

	/**
	 * Removes the specified ParameterListener object so that it no longer receives
	 * ParameterEvents.
	 *
	 * @param l the ParameterListener to register
	 * @see #addParameterListener(ParameterListener)
	 */
  public void removeParameterListener(ParameterListener l) {
    if (listeners == null) return;
    if ( l != null ) {
      listeners.remove( ParameterListener.class, l );
      }
    }

  public void fireDidChange(Number newValue, Number oldValue, Parameter source, boolean valueIsAdjusting) {
    if (listeners == null) return;
    Object[] lsns = listeners.getListenerList();
    ParameterEvent e = null;

    // Process the listeners last to first, notifying
    // those that are interested in this event
    for ( int i = lsns.length - 2; i >= 0; i -= 2 ) {
      if ( lsns[i] == ParameterListener.class ) {
        if (e == null)
          e = new ParameterEvent( source, newValue, oldValue, ParameterEvent.DID_CHANGE, valueIsAdjusting);
        ((ParameterListener)lsns[i+1]).didChange( e );
        }
      }
    }

  }
