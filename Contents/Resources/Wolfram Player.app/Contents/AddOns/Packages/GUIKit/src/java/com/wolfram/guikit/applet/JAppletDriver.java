/*
 * @(#)AppletDriver.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.applet;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Window;

import javax.swing.JApplet;
import javax.swing.JLabel;

import com.wolfram.guikit.*;

/**
 * AppletDriver
 *
 * Will it ever make sense for this to support/use the modal API, probably not
 */
public class JAppletDriver extends AppletDriver {

  public JAppletDriver(GUIKitDriverContainer container) {
    super(container);
    }
    
  public Object resolveExecuteObject(Object sourceObject) {
    Object executeObject = null;
    if (!(container instanceof JApplet)) return null;
    JApplet applet = (JApplet)container;
    
    // Do we add any specifics for JComponent or Frame or spawn external window
    // by default?
    
    if ((sourceObject instanceof Component)) {
      if (sourceObject instanceof Window) {
        executeObject = (Window)sourceObject;
        applet.getContentPane().add(new JLabel("GUIKit content is running in a separate window"), BorderLayout.CENTER);
        }
      else {
        applet.getContentPane().add((Component)sourceObject, BorderLayout.CENTER);
        executeObject = this;
        }
      }

    return executeObject;
    }

}
