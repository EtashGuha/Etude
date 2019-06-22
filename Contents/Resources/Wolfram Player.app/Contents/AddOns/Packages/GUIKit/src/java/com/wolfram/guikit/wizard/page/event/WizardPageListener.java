/*
 * @(#)WizardPageListener.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.wizard.page.event;

import java.util.EventListener;

/**
 * WizardPageListener
 *
 * @version $Revision: 1.1 $
 */
public interface WizardPageListener extends EventListener {

  public void pageWillActivate(WizardPageEvent e);
  public void pageDidActivate(WizardPageEvent e);
  public void pageWillDeactivate(WizardPageEvent e);
  public void pageDidDeactivate(WizardPageEvent e);
  
}