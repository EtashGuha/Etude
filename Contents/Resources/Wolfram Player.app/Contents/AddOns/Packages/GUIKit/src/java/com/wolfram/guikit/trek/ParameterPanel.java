/*
 * @(#)ParameterPanel.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.trek;

import java.awt.Component;

import javax.swing.*;

/**
 * ParameterPanel
 *
 * @version $Revision: 1.2 $
 */
public class ParameterPanel extends JPanel {

  private static final long serialVersionUID = -1283987975436738948L;
    
  public ParameterPanel() {
    super();
    setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
    }
  
  public void addGlue() {
    add(Box.createGlue());
    }
 
  public void addSpace(int val) {
    add(Box.createVerticalStrut(val));
    }

  public Component add(Component comp) {
    Component result = super.add(comp);
    if (!(comp instanceof Box.Filler))
      addSpace(8);
    return result;
   }

}