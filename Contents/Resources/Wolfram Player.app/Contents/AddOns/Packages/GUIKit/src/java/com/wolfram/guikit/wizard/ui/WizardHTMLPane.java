/*
 * @(#)WizardHTMLPane.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.wizard.ui;

import javax.swing.text.StyledDocument;

import com.wolfram.guikit.swing.GUIKitJHTMLPane;

/**
 * WizardHTMLPane extends GUIKitJTextPane
 */
public class WizardHTMLPane extends GUIKitJHTMLPane {
 
  private static final long serialVersionUID = -1287927972456728948L;
    
  private static boolean isMacOSX;
  
  static {
    try {
      String osName = System.getProperty("os.name");
      isMacOSX = osName != null && osName.toLowerCase().indexOf("mac") == 0;
      } 
    catch (SecurityException e) {}
    }
    
	public WizardHTMLPane() {
		super();
    init();
		}

  public WizardHTMLPane(StyledDocument doc) {
    super(doc);
    init();
    }
    
  protected void init() {
    super.init();
    setOpaque(false);
    setEditable(false);
    if (isMacOSX) setStyleSheet("BODY{font-family:Default;font-size:11;}");
    else setStyleSheet("BODY{font-family:sans-serif;font-size:11;}");
    }
  
  }
