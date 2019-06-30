/*
 * @(#)Wizard.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.wizard;

import java.awt.Container;
import java.awt.Paint;
import java.util.Collection;

import javax.swing.JDialog;

import com.wolfram.guikit.wizard.page.WizardPage;
import com.wolfram.guikit.wizard.ui.PageNavigationPanel;
import com.wolfram.guikit.wizard.ui.PageSideBarPanel;

/**
 * Wizard
 *
 * @version $Revision: 1.2 $
 */
public interface Wizard  {
 
  public static final int NAVIGATEBACK = 1;
  public static final int NAVIGATENEXT = 2;
  public static final int NAVIGATELAST = 4;
  public static final int NAVIGATECANCEL = 8;
  public static final int NAVIGATEHELP = 16;
  public static final int NAVIGATEFINISH = 32;
  public static final int NAVIGATECLOSE = 64;
  
  public WizardPage getPage(int index);
  
  public void reset();
  
  public WizardPage[] getPages();
  public void setPages(WizardPage[] pages);
  
  public void addPages(Collection c);
  public void addPage(WizardPage p);
  
  public WizardPage getNextPage(WizardPage page);
  public WizardPage getPreviousPage(WizardPage page);
  
  public WizardPage getLastPage();
  public void setLastPage(WizardPage p);
  
  public String getTitle();
  public void setTitle(String string);
  
  public WizardContainer getContainer();
  public void setContainer(WizardContainer wizardContainer);
  public WizardPage getCurrentPage();
    
  public PageSideBarPanel getSideBarPanel();
  public void setSideBarPanel(PageSideBarPanel panel);
  
  public Object getSideBarImage();
  public void setSideBarImage(Object image);
  
  public Paint getSideBarPaint();
  public void setSideBarPaint(Paint p);
  
  public Container getSideBarContent();
  public void setSideBarContent(Container c);
  
  public String getSideBarTitle();
  public void setSideBarTitle(String string);
  
  public JDialog getHelpDialog();
  public void setHelpDialog(JDialog dialog);
  
  public PageNavigationPanel getNavigationPanel();
  public void setNavigationPanel(PageNavigationPanel panel);
  
  public void setAllowBack(WizardPage page, boolean val);
  public void setAllowNext(WizardPage page, boolean val);
  public void setAllowLast(WizardPage page, boolean val);
  public void setAllowFinish(WizardPage page, boolean val);
  public void setAllowCancel(WizardPage page, boolean val);
  
  public void finish();
  public void close();
  public void cancel();
   
}
