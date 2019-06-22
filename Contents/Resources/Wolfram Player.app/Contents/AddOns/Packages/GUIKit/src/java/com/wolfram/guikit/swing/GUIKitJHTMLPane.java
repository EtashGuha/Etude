/*
 * @(#)GUIKitJHTMLPane.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing;

import java.io.IOException;
import java.net.URL;

import javax.swing.text.Document;
import javax.swing.text.StyledDocument;
import javax.swing.text.html.HTMLDocument;
import javax.swing.text.html.StyleSheet;

/**
 * GUIKitJHTMLPane extends GUIKitJTextPane
 */
public class GUIKitJHTMLPane extends GUIKitJTextPane {
 
  private static final long serialVersionUID = -1287947975456788448L;
    
  // TODO soon this needs to support styleSheet loading from
  // paths etc.. and not just in-memory stylesheet strings
  
  private String styleSheet = null;
  
	public GUIKitJHTMLPane() {
		super();
    init();
		}

  public GUIKitJHTMLPane(StyledDocument doc) {
    super(doc);
    init();
    }
    
  protected void init() {
    setContentType("text/html");
    }
  
  protected void applyStyleSheet() {
    if (styleSheet == null) return;
    Document doc = getDocument();
    if (doc != null && doc instanceof HTMLDocument) {
      StyleSheet ss = ((HTMLDocument)doc).getStyleSheet();
      if (ss != null) ss.addRule(styleSheet);
      }
    }
  
  public void setText(String t) {
    super.setText(t);
    applyStyleSheet();
    }
    
  public void setPage(URL page) throws IOException {
    super.setPage(page);
    applyStyleSheet();
    }
    
  public String getStyleSheet() {return styleSheet;}
  public void setStyleSheet(String s) {
    styleSheet = s;
    }
    
  }
