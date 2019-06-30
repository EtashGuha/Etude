/*
 * @(#)WizardDialog.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.wizard.ui;

import java.awt.Dialog;
import java.awt.Frame;

import com.wolfram.guikit.swing.GUIKitJDialog;

import com.wolfram.guikit.wizard.Wizard;
import com.wolfram.guikit.wizard.WizardContainer;
import com.wolfram.guikit.wizard.page.WizardPage;

/**
 * WizardDialog
 *
 * @version $Revision: 1.3 $
 */
public class WizardDialog extends GUIKitJDialog implements WizardContainer {

  private static final long serialVersionUID = -1287983975456388348L;
    
  private WizardContainerDelegate wizardContainerDelegate;
  
  public WizardDialog() {
    this((Frame)null, false);
    }

  public WizardDialog(Frame owner) {
    this(owner, false);
    }

  public WizardDialog(Frame owner, boolean modal){
    this(owner, null, modal);
    }

  public WizardDialog(Frame owner, String title) {
    this(owner, title, false);     
    }

  public WizardDialog(Dialog owner) {
    this(owner, false);
    }

  public WizardDialog(Dialog owner, boolean modal) {
    this(owner, null, modal);
    }

  public WizardDialog(Dialog owner, String title) {
    this(owner, title, false);     
    }

  public WizardDialog(Frame owner, String title, boolean modal) {
    super(owner, title, modal);
    init();
    }
    
  public WizardDialog(Dialog owner, String title, boolean modal) {
    super(owner, title, modal);
    init();
    }
  
  private void init() {
    wizardContainerDelegate = new WizardContainerDelegate(this);
    }
    
  public void dispose() {
    if (wizardContainerDelegate != null) wizardContainerDelegate.performClose(false);
    super.dispose();
    }

  // Delegated WizardContainer calls
  
	public void reset(){wizardContainerDelegate.reset();}
	
  public Wizard getWizard() {return wizardContainerDelegate.getWizard();}
  public void setWizard(Wizard w) {wizardContainerDelegate.setWizard(w);}
  
  public WizardPage getCurrentPage(){return wizardContainerDelegate.getCurrentPage();}
 
  public void showPage(WizardPage page){wizardContainerDelegate.showPage(page);}
  
  public void updateNavigationActions(){wizardContainerDelegate.updateNavigationActions();}
  
  public PageSideBarPanel getSideBarPanel(){return wizardContainerDelegate.getSideBarPanel();}
  public PageNavigationPanel getNavigationPanel(){return wizardContainerDelegate.getNavigationPanel();}
  public PageContentPanel getContentPanel(){return wizardContainerDelegate.getContentPanel();}

}
