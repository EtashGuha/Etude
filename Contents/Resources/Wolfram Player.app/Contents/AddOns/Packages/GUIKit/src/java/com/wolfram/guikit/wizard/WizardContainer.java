/*
 * @(#)WizardContainer.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.wizard;

import java.awt.Component;

import com.wolfram.guikit.wizard.page.WizardPage;
import com.wolfram.guikit.wizard.ui.PageContentPanel;
import com.wolfram.guikit.wizard.ui.PageNavigationPanel;
import com.wolfram.guikit.wizard.ui.PageSideBarPanel;

/**
 * WizardContainer
 *
 * @version $Revision: 1.1 $
 */
public interface WizardContainer  {
  
  public Wizard getWizard();
  public void setWizard(Wizard w);
  
  public WizardPage getCurrentPage();
 
  public void reset();
  public void showPage(WizardPage page);
  
  public void updateNavigationActions();
  
  public PageSideBarPanel getSideBarPanel();
  public PageNavigationPanel getNavigationPanel();
  public PageContentPanel getContentPanel();
  
  public String getTitle();
  public void setTitle(String title);
  
  public boolean isResizable();
  public void setResizable(boolean val);
  
  public boolean isAncestorOf(Component c);
  public void remove(Component comp);
  public void removeAll();
  
}
