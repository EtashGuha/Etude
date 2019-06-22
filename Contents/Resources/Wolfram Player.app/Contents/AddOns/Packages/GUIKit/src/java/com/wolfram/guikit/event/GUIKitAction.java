/*
 * @(#)GUIKitAction.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.event;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.AbstractAction;
import javax.swing.Icon;
import javax.swing.KeyStroke;
import javax.swing.event.EventListenerList;

/**
 * GUIKitAction
 */
public class GUIKitAction extends AbstractAction {
  
  private static final long serialVersionUID = -1687987965456686948L;
    
  protected EventListenerList listeners = null;
  
  public GUIKitAction() {
    super();
    }
  
  public GUIKitAction(String name) {
    super(name);
    }

  public GUIKitAction(String name, Icon icon) {
    super(name, icon);
    }

  /**
   * Adds the specified SwingPaintListener to receive SwingPaintEvents.
   * <p>
   * Use this method to register a SwingPaintListener object to receive
   * notifications when painting occurs on this component
   *
   * @param l the SwingPaintListener to register
   * @see #removePaintListener(SwingPaintListener)
   */
  public void addActionListener(ActionListener l) {
    if (listeners == null) listeners = new EventListenerList();
    if ( l != null ) {
      listeners.add( ActionListener.class, l );
      }
    }

  /**
   * Removes the specified SwingPaintListener object so that it no longer receives
   * SwingPaintEvents.
   *
   * @param l the SwingPaintListener to register
   * @see #addPaintListener(SwingPaintListener)
   */
  public void removeActionListener(ActionListener l) {
    if (listeners == null) return;
    if ( l != null ) {
      listeners.remove( ActionListener.class, l );
      }
    }
    
  public void actionPerformed(ActionEvent e) {
    if (listeners == null) return;
    Object[] lsns = listeners.getListenerList();

    // Process the listeners last to first, notifying
    // those that are interested in this event
    for ( int i = lsns.length - 2; i >= 0; i -= 2 ) {
      if ( lsns[i] == ActionListener.class ) {
        ((ActionListener)lsns[i+1]).actionPerformed(e);
        }
      }
    }
 
  // We want to support all the basic Action keys
  // as properties
  public String getName(){
    Object result = getValue(NAME);
    if(result != null && result instanceof String) return (String)result;
    return null;
    }
  public void setName(String name) {
    putValue(NAME, name);
    }

  public String getCommandKey(){
    Object result = getValue(ACTION_COMMAND_KEY);
    if(result != null && result instanceof String) return (String)result;
    return null;
    }
  public void setCommandKey(String name) {
    putValue(ACTION_COMMAND_KEY, name);
    }

  public String getShortDescription(){
    Object result = getValue(SHORT_DESCRIPTION);
    if(result != null && result instanceof String) return (String)result;
    return null;
    }
  public void setShortDescription(String name) {
    putValue(SHORT_DESCRIPTION, name);
    }

  public String getLongDescription(){
    Object result = getValue(LONG_DESCRIPTION);
    if(result != null && result instanceof String) return (String)result;
    return null;
    }
  public void setLongDescription(String name) {
    putValue(LONG_DESCRIPTION, name);
    }
    
  public Icon getIcon(){
    return getSmallIcon();
    }
  public void setIcon(Icon i) {
    setSmallIcon(i);
    }
    
  public Icon getSmallIcon(){
    Object result = getValue(SMALL_ICON);
    if(result != null && result instanceof Icon) return (Icon)result;
    return null;
    }
  public void setSmallIcon(Icon i) {
    putValue(SMALL_ICON, i);
    }
    
  public KeyStroke getAccelerator(){
    Object result = getValue(ACCELERATOR_KEY);
    if(result != null && result instanceof KeyStroke) return (KeyStroke)result;
    return null;
    }
  public void setAccelerator(KeyStroke k) {
    putValue(ACCELERATOR_KEY, k);
    }
    
  public Integer getMnemonic(){
    Object result = getValue(MNEMONIC_KEY);
    if(result != null && result instanceof Integer) return (Integer)result;
    return null;
    }
  public void setMnemonic(Integer i) {
    putValue(MNEMONIC_KEY, i);
    }
    
}
