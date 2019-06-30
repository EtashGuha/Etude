/*
 * @(#)AWTPaintEventHandler.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.awt;

import java.awt.Graphics;

import javax.swing.event.EventListenerList;

/**
 * AWTPaintEventHandler is a utility class that wraps all the code needed to process
 * and manage AWT paint events and the listeners.
 * <p>
 * Use of this class simplifies the code needed in subclasses of AWT components
 * that want to implement AWTPaintEvents
 */
public class AWTPaintEventHandler  {

  protected EventListenerList listeners = null;

	/**
	 * Adds the specified AWTPaintListener to receive AWTPaintEvents.
	 * <p>
	 * Use this method to register an AWTPaintListener object to receive
	 * notifications when painting occurs on this component
	 *
	 * @param l the AWTPaintListener to register
	 * @see #removePaintListener(AWTPaintListener)
	 */
  public void addPaintListener(AWTPaintListener l) {
    if (listeners == null) listeners = new EventListenerList();
    if ( l != null ) {
      listeners.add( AWTPaintListener.class, l );
      }
    }

	/**
	 * Removes the specified AWTPaintListener object so that it no longer receives
	 * AWTPaintEvents.
	 *
	 * @param l the AWTPaintListener to register
	 * @see #addPaintListener(AWTPaintListener)
	 */
  public void removePaintListener(AWTPaintListener l) {
    if (listeners == null) return;
    if ( l != null ) {
      listeners.remove( AWTPaintListener.class, l );
      }
    }

  public void fireWillPaint(Graphics g, Object source) {
    if (listeners == null) return;
    Object[] lsns = listeners.getListenerList();
    AWTPaintEvent e = null;

    // Process the listeners last to first, notifying
    // those that are interested in this event
    for ( int i = lsns.length - 2; i >= 0; i -= 2 ) {
      if ( lsns[i] == AWTPaintListener.class ) {
        if (e == null)
          e = new AWTPaintEvent( source, g, AWTPaintEvent.WILL_PAINT );
        ((AWTPaintListener)lsns[i+1]).willPaint( e );
        }
      }
    }

  public void fireDidPaint(Graphics g, Object source) {
    if (listeners == null) return;
    Object[] lsns = listeners.getListenerList();
    AWTPaintEvent e = null;

    // Process the listeners last to first, notifying
    // those that are interested in this event
    for ( int i = lsns.length - 2; i >= 0; i -= 2 ) {
      if ( lsns[i] == AWTPaintListener.class ) {
        if (e == null)
          e = new AWTPaintEvent( source, g, AWTPaintEvent.DID_PAINT );
        ((AWTPaintListener)lsns[i+1]).didPaint( e );
        }
      }
    }

  public void fireWillUpdate(Graphics g, Object source) {
    if (listeners == null) return;
    Object[] lsns = listeners.getListenerList();
    AWTPaintEvent e = null;

    // Process the listeners last to first, notifying
    // those that are interested in this event
    for ( int i = lsns.length - 2; i >= 0; i -= 2 ) {
      if ( lsns[i] == AWTPaintListener.class ) {
        if (e == null)
          e = new AWTPaintEvent( source, g, AWTPaintEvent.WILL_UPDATE );
        ((AWTPaintListener)lsns[i+1]).willUpdate( e );
        }
      }
    }

  public void fireDidUpdate(Graphics g, Object source) {
    if (listeners == null) return;
    Object[] lsns = listeners.getListenerList();
    AWTPaintEvent e = null;

    // Process the listeners last to first, notifying
    // those that are interested in this event
    for ( int i = lsns.length - 2; i >= 0; i -= 2 ) {
      if ( lsns[i] == AWTPaintListener.class ) {
        if (e == null)
          e = new AWTPaintEvent( source, g, AWTPaintEvent.DID_UPDATE );
        ((AWTPaintListener)lsns[i+1]).didUpdate( e );
        }
      }
    }

  }
