/*
 * @(#)GUIKitJFrame.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing;

import java.awt.GraphicsConfiguration;
import java.awt.Image;

import javax.swing.JFrame;
import javax.swing.JMenuBar;

import com.wolfram.guikit.GUIKitDriver;
import com.wolfram.guikit.util.WindowUtils;

/**
 * GUIKitJFrame is a subclass of JFrame
 * used in GUIKit to add useful properties and abstractions
 */
public class GUIKitJFrame extends JFrame {
 
  private static final long serialVersionUID = -1287987975456738943L;
    
  public GUIKitJFrame() {
    super();
    initGUI();      
    }

  public GUIKitJFrame(GraphicsConfiguration gc) {
    super(gc);
    initGUI();
    }

  public GUIKitJFrame(String title) {
    super(title);
    initGUI();
    }

  public GUIKitJFrame(String title, GraphicsConfiguration gc) {
    super(title, gc);
    initGUI();
    }
  
  private void initGUI() {
    Image img = GUIKitDriver.getDefaultFrameImage();
    if (img != null) setIconImage(img);
    }
  
  public void setMenus(JMenuBar menubar) {
    setJMenuBar(menubar);
    }

  public JMenuBar getMenus() { 
    return getJMenuBar(); 
    }
    
  public void center() {
    WindowUtils.centerComponent(this);
    }
    
  }
