/*
 * @(#)GUIKitJInternalFrame.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing;

import javax.swing.JInternalFrame;
import javax.swing.JMenuBar;

/**
 * GUIKitJInternalFrame is a subclass of JInternalFrame
 * used in GUIKit to add useful properties and abstractions
 */
public class GUIKitJInternalFrame extends JInternalFrame {

  private static final long serialVersionUID = -1282987977456787948L;
    
  public GUIKitJInternalFrame() {
    this("", false, false, false, false);
    }

  public GUIKitJInternalFrame(String title) {
    this(title, false, false, false, false);
    }

  public GUIKitJInternalFrame(String title, boolean resizable) {
    this(title, resizable, false, false, false);
    }

  public GUIKitJInternalFrame(String title, boolean resizable, boolean closable) {
    this(title, resizable, closable, false, false);
    }

  public GUIKitJInternalFrame(String title, boolean resizable, boolean closable, boolean maximizable) {
    this(title, resizable, closable, maximizable, false);
    }

  public GUIKitJInternalFrame(String title, boolean resizable, boolean closable, boolean maximizable, boolean iconifiable) {
    super(title, resizable, closable, maximizable, iconifiable);
    }
      
  public void setMenus(JMenuBar menubar) {
    setJMenuBar(menubar);
    }

  public JMenuBar getMenus() { 
    return getJMenuBar(); 
    }
    
  }
