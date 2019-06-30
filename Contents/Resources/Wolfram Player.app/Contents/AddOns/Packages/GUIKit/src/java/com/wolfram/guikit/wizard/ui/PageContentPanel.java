/*
 * @(#)PageContentPanel.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.wizard.ui;

import java.awt.BorderLayout;
import java.awt.Container;

import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSeparator;
import javax.swing.border.EmptyBorder;

/**
 * PageContentPanel
 *
 * @version $Revision: 1.2 $
 */
public class PageContentPanel extends JPanel {
   
  private static final long serialVersionUID = -1287887975456282948L;
    
  private String title;
  private Container content;
 
  private JPanel centerPanel;
  
  private JLabel titleLabel;
  
  public PageContentPanel() {
    this(true);
    }
    
  public PageContentPanel(boolean isDoubleBuffered) {
    super(new BorderLayout(), isDoubleBuffered);
    
    setBorder(new EmptyBorder(11, 10, 5, 5));
    
    titleLabel = new JLabel("Title");
    
    JPanel titlePanel = new JPanel();
    titlePanel.setLayout(new BoxLayout(titlePanel, BoxLayout.Y_AXIS));
    titlePanel.add(titleLabel);
    titlePanel.add(Box.createVerticalStrut(5));
    titlePanel.add(new JSeparator(JSeparator.HORIZONTAL));
    titlePanel.add(Box.createVerticalStrut(11));
    titlePanel.add(Box.createGlue());
    
    add(titlePanel, BorderLayout.NORTH);
    
    centerPanel = new JPanel();
    centerPanel.setLayout(new BoxLayout(centerPanel, BoxLayout.Y_AXIS));
    centerPanel.add(Box.createGlue());
    
    add(centerPanel, BorderLayout.CENTER);
    }
    
  // This is the area where custom user content goes within a wizard page
  public Container getContent() {return content;}
  public void setContent(Container panel) {
    Container oldPanel = content;
    content = panel;
    
    if ((oldPanel == content) && isAncestorOf(content)) return;
    
    if (oldPanel != content) {
      if (oldPanel != null) {
        if (isAncestorOf(oldPanel)) remove(oldPanel);
        }
      }
    
    centerPanel.removeAll();
    if (content != null && !isAncestorOf(content)) {
      centerPanel.add(content);
      }
    centerPanel.add(Box.createGlue());
    }
  
  public String getTitle() {return title;}
  public void setTitle(String string){
    title = string;
    if (titleLabel != null) titleLabel.setText((title != null) ? title : " ");
    }
 
}