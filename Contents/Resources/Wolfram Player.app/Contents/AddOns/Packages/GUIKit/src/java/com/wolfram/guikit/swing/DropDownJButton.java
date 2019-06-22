/*
 * @(#)DropDownJButton.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;

import javax.swing.Action;
import javax.swing.DefaultButtonModel;
import javax.swing.Icon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JMenuItem;
import javax.swing.JPopupMenu;
import javax.swing.UIManager;


/**
 * DropDownJButton provides a subclass of JButton that includes
 * a drop down popup menu
 */
public class DropDownJButton extends JButton {
 
  private static final long serialVersionUID = -1287987955456588548L;
  
  protected JPopupMenu popupMenu;
  protected boolean includeDropDownGraphics = true;
  
  public DropDownJButton() {
    this(null, null);
    }
  
  public DropDownJButton(Icon icon) {
    this(null, icon);
    }
  
  public DropDownJButton(String text) {
    this(text, null);
    }
  
  public DropDownJButton(Action a) {
    this();
    setAction(a);
    }

  public DropDownJButton(String text, Icon icon) {
    // Create the model
    setModel(new DefaultButtonModel());

    // initialize
    init(text, icon);
    
    addMouseListener(new MouseAdapter() {
      public void mouseClicked(MouseEvent e) {
        if (popupMenu != null && e.getClickCount() >= 2 && !popupMenu.isVisible()) {
          popupMenu.show(DropDownJButton.this, 0, getHeight());
          }
        }
      });
    addMouseMotionListener(new MouseMotionAdapter() {
      public void mouseDragged(MouseEvent e) {
        if (popupMenu != null && !popupMenu.isVisible()) {
          popupMenu.show(DropDownJButton.this, 0, getHeight());
          }
        }
      });
    }
    
  public JPopupMenu getPopupMenu() {return popupMenu;}
  
  public void setPopupMenu(JPopupMenu newMenu) {
    popupMenu = newMenu;
    }
  
  public boolean getIncludeDropDownGraphics() {return includeDropDownGraphics;}
  public void setIncludeDropDownGraphics(boolean val) {
    includeDropDownGraphics = val;
    repaint();
    }
    
  protected void paintComponent(Graphics g) {
    super.paintComponent(g);
    
    if (popupMenu != null && includeDropDownGraphics) {
      int originx = getWidth() - 10;
      int originy = getHeight()/2 + 1;
      g.setColor(Color.BLACK);
      g.drawLine(originx - 2, originy - 2, originx + 3, originy - 2);
      g.drawLine(originx - 1, originy - 1, originx + 2, originy - 1);
      g.drawLine(originx, originy, originx + 1, originy);
      }
    }
   
  // exists for testing purposes only
  public static void main(String args[]) {
    
    try {
      UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
    }
    catch (Exception e){}
    

    JFrame frame = new JFrame();
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    
    DropDownJButton b = new DropDownJButton("Test");
    JPopupMenu p = new JPopupMenu();
    p.add(new JMenuItem("Points"));
    p.add(new JMenuItem("Lines"));
    b.setPopupMenu(p);
    
    frame.getContentPane().add(b);
    frame.pack();
    frame.show();

    }
    
  }
