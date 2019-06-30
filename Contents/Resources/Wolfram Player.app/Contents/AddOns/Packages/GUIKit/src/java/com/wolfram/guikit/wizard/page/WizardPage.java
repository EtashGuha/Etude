/*
 * @(#)Wizard.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.wizard.page;

import java.awt.Container;
import java.awt.Paint;

import com.wolfram.guikit.wizard.Wizard;
import com.wolfram.guikit.wizard.page.event.WizardPageListener;
import com.wolfram.guikit.wizard.ui.*;

/**
 * WizardPage
 *
 * @version $Revision: 1.1 $
 */
public interface WizardPage  {
  
  public Wizard getWizard();
  public void setWizard(Wizard w);
  
  public WizardPage getNextPage();
  public void setNextPage(WizardPage page);
  
  public WizardPage getPreviousPage();
  public void setPreviousPage(WizardPage page);
  
  public PageContentPanel getContentPanel();
  public void setContentPanel(PageContentPanel panel);
  
  public int getNavigationMask();
  public void setNavigationMask(int mask);
  
  public String[] getNavigationNames();
  public void setNavigationNames(String[] names);
  
  // This is the area where custom user content goes within a wizard page.
  // Normally individual pages will place custom content here and share
  // a contentPanel
  public Container getContent();
  public void setContent(Container inputPanel);
  
  public PageNavigationPanel getNavigationPanel();
  public void setNavigationPanel(PageNavigationPanel panel);
  
  public PageSideBarPanel getSideBarPanel();
  public void setSideBarPanel(PageSideBarPanel panel);
  
  public String getSideBarTitle();
  public void setSideBarTitle(String string);
  
  public Object getSideBarImage();
  public void setSideBarImage(Object image);
  
  public Paint getSideBarPaint();
  public void setSideBarPaint(Paint p);
    
  public Container getSideBarContent();
  public void setSideBarContent(Container c);
  
  // This should probably be a convenience wrapper for getContentPanel().getTitle()
  public String getTitle();
  public void setTitle(String string);
  
  // Need booleans for allowNext, allowFinish etc
  // each page will be checked as well as wizard for determining any vetoers
  public boolean getAllowBack();
  public void setAllowBack(boolean val);
  public boolean getAllowNext();
  public void setAllowNext(boolean val);
  public boolean getAllowFinish();
  public void setAllowFinish(boolean val);
  public boolean getAllowLast();
  public void setAllowLast(boolean val);
  public boolean getAllowCancel();
  public void setAllowCancel(boolean val);
  
  public boolean isActive();
  public void setActive(boolean val);
  
  public void addWizardPageListener(WizardPageListener l);
  public void removeWizardPageListener(WizardPageListener l);
  public void fireWillActivate();
  public void fireWillDeactivate();
    
}