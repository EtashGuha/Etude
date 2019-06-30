/*
 * @(#)PageSideBarPanel.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.wizard.ui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Container;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.Paint;

import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSeparator;
import javax.swing.border.EmptyBorder;

import com.wolfram.guikit.swing.GUIKitImageJLabel;

/**
 * PageSideBarPanel
 *
 * @version $Revision: 1.2 $
 */
public class PageSideBarPanel extends JPanel {

  private static final long serialVersionUID = -1287982975452782948L;
    
  private Paint sideBarPaint;
  
  private Object sideBarImage;
  private Image useSideBarImage;
  
  private Container sideBarContent;
  private String sideBarTitle;
  private JLabel titleLabel = new JLabel(" ");
  
  public PageSideBarPanel() {
    this(true);
    }
    
  public PageSideBarPanel(boolean isDoubleBuffered) {
    super(new BorderLayout(), isDoubleBuffered);
    setBackground(Color.WHITE);
    doSideBarLayout();
    }
    
  public void doSideBarLayout() {
    removeAll();
    if (sideBarContent != null)
      layoutSideBarContent();
    }
  
  public void paintComponent(Graphics g) {
    super.paintComponent(g);
    if (sideBarPaint != null) {
      Graphics2D g2 = (Graphics2D)g;
      Paint oldPaint = g2.getPaint();
      g2.setPaint(sideBarPaint);
      g2.fillRect(0, 0, getWidth(), getHeight());
      g2.setPaint(oldPaint);
      }
    if (useSideBarImage != null)
      g.drawImage(useSideBarImage, 0, 0, this);
    }
  

  protected void layoutSideBarContent() {
     setBorder(new EmptyBorder(11,11,5,10));

    titleLabel.setOpaque(false);
    
    JPanel titlePanel = new JPanel();
    titlePanel.setOpaque(false);
    titlePanel.setLayout(new BoxLayout(titlePanel, BoxLayout.Y_AXIS));
    titlePanel.add(titleLabel);
    titlePanel.add(Box.createVerticalStrut(5));
    titlePanel.add(new JSeparator(JSeparator.HORIZONTAL));
    titlePanel.add(Box.createVerticalStrut(11));
    titlePanel.add(Box.createGlue());
    
    add(titlePanel, BorderLayout.NORTH);
    
    JPanel centerPanel = new JPanel();
    centerPanel.setOpaque(false);
    centerPanel.setLayout(new BoxLayout(centerPanel, BoxLayout.Y_AXIS));
    // Note we may need to always set sideBarContent component not-opaque
    // to see sidebar background
    if (sideBarContent instanceof JComponent) ((JComponent)sideBarContent).setOpaque(false);
    centerPanel.add(sideBarContent);

    centerPanel.add(Box.createGlue());
    
    add(centerPanel, BorderLayout.CENTER);
    }
    
  public Paint getSideBarPaint() {return sideBarPaint;}
  public void setSideBarPaint(Paint p){
    sideBarPaint = p;
    }
    
  public Object getSideBarImage() {return sideBarImage;}
  public void setSideBarImage(Object i){
    sideBarImage = i;
    useSideBarImage = null;
    if (sideBarImage instanceof ImageIcon) 
      useSideBarImage = ((ImageIcon)sideBarImage).getImage();
    else if (sideBarImage instanceof GUIKitImageJLabel) 
      useSideBarImage = ((ImageIcon)((GUIKitImageJLabel)sideBarImage).getIcon()).getImage();
    else if (sideBarImage instanceof Image) 
      useSideBarImage = ((Image)sideBarImage);
    }

  public Container getSideBarContent() {return sideBarContent;}
  public void setSideBarContent(Container c) {
    sideBarContent = c;
    }
    
  public String getSideBarTitle() {return sideBarTitle;}
  public void setSideBarTitle(String t) {
    sideBarTitle = t;
        
    if (sideBarTitle != null) titleLabel.setText(sideBarTitle);
    else titleLabel.setText(" ");
    }
    
}