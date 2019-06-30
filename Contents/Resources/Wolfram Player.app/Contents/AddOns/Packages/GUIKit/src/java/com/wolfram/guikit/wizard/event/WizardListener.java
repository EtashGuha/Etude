/*
 * @(#)Wizard.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.wizard.event;

import java.util.EventListener;

/**
 * Wizard
 *
 * @version $Revision: 1.2 $
 */
public interface WizardListener extends EventListener {
 
  /** Successful ending with possible result */
  public void wizardFinished(WizardEvent e);
  
  /** Unsuccessful ending with no possible result */
  public void wizardCanceled(WizardEvent e);
  
  /** Closing of resources and window after a finish or cancel */
  public void wizardClosed(WizardEvent e);
 
  /** Called if a live wizard was reset to its initial state */
  public void wizardDidReset(WizardEvent e);
  
}