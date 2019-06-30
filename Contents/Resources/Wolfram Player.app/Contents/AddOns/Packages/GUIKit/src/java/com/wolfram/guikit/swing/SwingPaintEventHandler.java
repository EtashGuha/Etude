/*
 * @(#)SwingPaintEventHandler.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing;

import java.awt.Graphics;

import javax.swing.event.EventListenerList;

/**
 * SwingPaintEventHandler is a utility class that wraps all the code needed to process
 * and manage Swing paint events and the listeners.
 * <p>
 * Use of this class simplifies the code needed in subclasses of Swing components
 * that want to implement SwingPaintEvents
 */
public class SwingPaintEventHandler  {

  protected EventListenerList listeners = null;

	/**
	 * Adds the specified SwingPaintListener to receive SwingPaintEvents.
	 * <p>
	 * Use this method to register a SwingPaintListener object to receive
	 * notifications when painting occurs on this component
	 *
	 * @param l the SwingPaintListener to register
	 * @see #removePaintListener(SwingPaintListener)
	 */
  public void addPaintListener(SwingPaintListener l) {
    if (listeners == null) listeners = new EventListenerList();
    if ( l != null ) {
      listeners.add( SwingPaintListener.class, l );
      }
    }


	/**
	 * Removes the specified SwingPaintListener object so that it no longer receives
	 * SwingPaintEvents.
	 *
	 * @param l the SwingPaintListener to register
	 * @see #addPaintListener(SwingPaintListener)
	 */
  public void removePaintListener(SwingPaintListener l) {
    if (listeners == null) return;
    if ( l != null ) {
      listeners.remove( SwingPaintListener.class, l );
      }
    }

  public void fireWillPaintComponent(Graphics g, Object source) {
    if (listeners == null) return;
    Object[] lsns = listeners.getListenerList();
    SwingPaintEvent e = null;

    // Process the listeners last to first, notifying
    // those that are interested in this event
    for ( int i = lsns.length - 2; i >= 0; i -= 2 ) {
      if ( lsns[i] == SwingPaintListener.class ) {
        if (e == null)
          e = new SwingPaintEvent( source, g, SwingPaintEvent.WILL_PAINT_COMPONENT );
        ((SwingPaintListener)lsns[i+1]).willPaintComponent( e );
        }
      }
    }

  public void fireDidPaintComponent(Graphics g, Object source) {
    if (listeners == null) return;
    Object[] lsns = listeners.getListenerList();
    SwingPaintEvent e = null;

    // Process the listeners last to first, notifying
    // those that are interested in this event
    for ( int i = lsns.length - 2; i >= 0; i -= 2 ) {
      if ( lsns[i] == SwingPaintListener.class ) {
        if (e == null)
          e = new SwingPaintEvent( source, g, SwingPaintEvent.DID_PAINT_COMPONENT );
        ((SwingPaintListener)lsns[i+1]).didPaintComponent( e );
        }
      }
    }

  public void fireWillPaintBorder(Graphics g, Object source) {
    if (listeners == null) return;
    Object[] lsns = listeners.getListenerList();
    SwingPaintEvent e = null;

    // Process the listeners last to first, notifying
    // those that are interested in this event
    for ( int i = lsns.length - 2; i >= 0; i -= 2 ) {
      if ( lsns[i] == SwingPaintListener.class ) {
        if (e == null)
          e = new SwingPaintEvent( source, g, SwingPaintEvent.WILL_PAINT_BORDER );
        ((SwingPaintListener)lsns[i+1]).willPaintBorder( e );
        }
      }
    }

  public void fireDidPaintBorder(Graphics g, Object source) {
    if (listeners == null) return;
    Object[] lsns = listeners.getListenerList();
    SwingPaintEvent e = null;

    // Process the listeners last to first, notifying
    // those that are interested in this event
    for ( int i = lsns.length - 2; i >= 0; i -= 2 ) {
      if ( lsns[i] == SwingPaintListener.class ) {
        if (e == null)
          e = new SwingPaintEvent( source, g, SwingPaintEvent.DID_PAINT_BORDER );
        ((SwingPaintListener)lsns[i+1]).didPaintBorder( e );
        }
      }
    }

  public void fireWillPaintChildren(Graphics g, Object source) {
    if (listeners == null) return;
    Object[] lsns = listeners.getListenerList();
    SwingPaintEvent e = null;

    // Process the listeners last to first, notifying
    // those that are interested in this event
    for ( int i = lsns.length - 2; i >= 0; i -= 2 ) {
      if ( lsns[i] == SwingPaintListener.class ) {
        if (e == null)
          e = new SwingPaintEvent( source, g, SwingPaintEvent.WILL_PAINT_CHILDREN );
        ((SwingPaintListener)lsns[i+1]).willPaintChildren( e );
        }
      }
    }

  public void fireDidPaintChildren(Graphics g, Object source) {
    if (listeners == null) return;
    Object[] lsns = listeners.getListenerList();
    SwingPaintEvent e = null;

    // Process the listeners last to first, notifying
    // those that are interested in this event
    for ( int i = lsns.length - 2; i >= 0; i -= 2 ) {
      if ( lsns[i] == SwingPaintListener.class ) {
        if (e == null)
          e = new SwingPaintEvent( source, g, SwingPaintEvent.DID_PAINT_CHILDREN );
        ((SwingPaintListener)lsns[i+1]).didPaintChildren( e );
        }
      }
    }

  }
