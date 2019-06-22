/*
 * @(#)GUIKitJDialog.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing;

import java.awt.Dialog;
import java.awt.Frame;
import java.awt.GraphicsConfiguration;
import java.awt.Image;
import java.awt.Window;

import javax.swing.JDialog;

import com.wolfram.guikit.util.WindowUtils;

/**
 * GUIKitJDialog is a subclass of JDialog
 * used in GUIKit to add useful properties and abstractions
 */
public class GUIKitJDialog extends JDialog {
   
  private static final long serialVersionUID = -1287937975456738938L;
   
  private static GUIKitJFrame sharedFrame = null;
  
  public static void setSharedFrameIconImage(Image im) {
    if (sharedFrame == null) {
      sharedFrame = new GUIKitJFrame();
      }
    sharedFrame.setIconImage(im);
    }
  
  public GUIKitJDialog() {
    this(sharedFrame, false);
    }

  public GUIKitJDialog(Frame owner) {
    this(owner, false);
    }

  public GUIKitJDialog(Frame owner, boolean modal) {
    this(owner, null, modal);
    }

  public GUIKitJDialog(Frame owner, String title) {
    this(owner, title, false);     
    }

  public GUIKitJDialog(Frame owner, String title, boolean modal) {
    super(owner, title, modal);
    }

  public GUIKitJDialog(Frame owner, String title, boolean modal, GraphicsConfiguration gc) {
    super(owner, title, modal, gc);
    }

  public GUIKitJDialog(Dialog owner) {
    this(owner, false);
    }

  public GUIKitJDialog(Dialog owner, boolean modal) {
    this(owner, null, modal);
    }

  public GUIKitJDialog(Dialog owner, String title)  {
    this(owner, title, false);     
    }

  public GUIKitJDialog(Dialog owner, String title, boolean modal) {
    super(owner, title, modal);
    }

  public GUIKitJDialog(Dialog owner, String title, boolean modal, GraphicsConfiguration gc) {
    super(owner, title, modal, gc);
    }
    
  public void setIconImage(Image image) {
    Window w = getOwner();
    if (w != null && w instanceof Frame) {
      ((Frame)w).setIconImage(image);
      }
    }
       
  public Image getIconImage() {
    Window w = getOwner();
    if (w != null && w instanceof Frame) {
      return ((Frame)w).getIconImage();
      }
    else return null;
    }
    
  public void center() {
    WindowUtils.centerComponent(this);
    }
    
  }
