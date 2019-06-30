/*
 * @(#)WizardFrame.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.wizard.ui;

import java.awt.GraphicsConfiguration;

import com.wolfram.guikit.swing.GUIKitJFrame;

import com.wolfram.guikit.wizard.Wizard;
import com.wolfram.guikit.wizard.WizardContainer;
import com.wolfram.guikit.wizard.page.WizardPage;

/**
 * WizardFrame
 *
 * @version $Revision: 1.3 $
 */
public class WizardFrame extends GUIKitJFrame implements WizardContainer {

  private static final long serialVersionUID = -1287487975458788448L;
    
  private WizardContainerDelegate wizardContainerDelegate;
  
  public WizardFrame() {
    super();
    init();
    }
  public WizardFrame(GraphicsConfiguration gc) {
    super(gc);
    init();
    }
  public WizardFrame(String title) {
    super(title);
    init();
    }
  public WizardFrame(String title, GraphicsConfiguration gc) {
    super(title, gc);
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
