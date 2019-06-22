/*
 * @(#)PageNavigationPanel.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.wizard.ui;

import java.net.URL;

import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JPanel;
import javax.swing.border.EmptyBorder;

import com.wolfram.guikit.wizard.Wizard;
import com.wolfram.guikit.wizard.WizardUtils;


/**
 * NavigationButtonsPanel
 *
 * @version $Revision: 1.2 $
 */
public class PageNavigationPanel extends JPanel {

  private static final long serialVersionUID = -1207987975056708948L;
    
  private JButton backButton;
  private JButton nextButton;
  private JButton lastButton;
  private JButton finishButton;
  private JButton cancelButton;
  private JButton closeButton;
  private JButton helpButton;
  
  private int navigationMask = 0;
  
  public PageNavigationPanel() {
    this(0);
    }

  public PageNavigationPanel(int mask) {
    super();
    init();
    setNavigationMask(mask);
    }

  public PageNavigationPanel(String[] buttonNames) {
    super();
    init();
    setNavigationNames(buttonNames);
    }
    
  public int getNavigationMask() {return navigationMask;}
  public void setNavigationMask(int mask) {
    if (mask == navigationMask) return;
    navigationMask = mask;
    updateButtonsLayout();
    }
    
  public String[] getNavigationNames() {
    return WizardUtils.getNavigationNames(navigationMask);
    }
  public void setNavigationNames(String[] names) {
    setNavigationMask(WizardUtils.getNavigationMask(names));
    }
  
  public JButton getBackButton() {return backButton;}
  public JButton getNextButton() {return nextButton;}
  public JButton getLastButton() {return lastButton;}
  public JButton getFinishButton() {return finishButton;}
  public JButton getCancelButton() {return cancelButton;}
  public JButton getCloseButton() {return closeButton;}
  public JButton getHelpButton() {return helpButton;}
  
  private void init() {
    setLayout(new BoxLayout(this, BoxLayout.X_AXIS));
    setBorder(new EmptyBorder(11,0,11,0));
    createButtons();
    }
  
  private void createButtons() {
    ImageIcon ic = null;
    URL imageURL;
    
    backButton = new JButton("Back");
    imageURL = PageNavigationPanel.class.getClassLoader().getResource(
      "images/wizard/Back16.gif");
    if (imageURL != null) {
      ic = new ImageIcon(imageURL);
      backButton.setIcon(ic);
      backButton.setHorizontalTextPosition(JButton.TRAILING);
      }
    nextButton = new JButton("Next");
    imageURL = PageNavigationPanel.class.getClassLoader().getResource(
      "images/wizard/Next16.gif");
    if (imageURL != null) {
      ic = new ImageIcon(imageURL);
      nextButton.setIcon(ic);
      nextButton.setHorizontalTextPosition(JButton.LEADING);
      }
    lastButton = new JButton("Last");
    finishButton = new JButton("Finish");
    cancelButton = new JButton("Cancel");
    closeButton = new JButton("Close");
    helpButton = new JButton("Help");
    }
    
 private void updateButtonsLayout() {
    removeAll();

    boolean addedOne = false;
    
    if ((navigationMask & Wizard.NAVIGATEBACK) != 0) {
      add(backButton);
      addedOne = true;
      }
    if ((navigationMask & Wizard.NAVIGATENEXT) != 0) {
      if (addedOne) add(Box.createHorizontalStrut(4));
      add(nextButton);
      addedOne = true;
      }
    if ((navigationMask & Wizard.NAVIGATELAST) != 0) {
      if (addedOne) add(Box.createHorizontalStrut(4));
      add(lastButton);
      addedOne = true;
      }
    if ((navigationMask & Wizard.NAVIGATEFINISH) != 0) {
      if (addedOne) add(Box.createHorizontalStrut(4));
      add(finishButton);
      addedOne = true;
      }
      
    add(Box.createGlue());
    
    addedOne = false;
    
    if ((navigationMask & Wizard.NAVIGATECANCEL) != 0) {
      add(cancelButton);
      addedOne = true;
      }
    if ((navigationMask & Wizard.NAVIGATECLOSE) != 0) {
      if (addedOne) add(Box.createHorizontalStrut(4));
      add(closeButton);
      addedOne = true;
      }
    if ((navigationMask & Wizard.NAVIGATEHELP) != 0) {
      if (addedOne) add(Box.createHorizontalStrut(4));
      add(helpButton);
      addedOne = true;
      }
       
    if (addedOne) add(Box.createHorizontalStrut(11));
    
    doLayout();
    }
  
}
