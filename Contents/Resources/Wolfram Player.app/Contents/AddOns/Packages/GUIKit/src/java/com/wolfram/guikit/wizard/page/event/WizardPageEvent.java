/*
 * @(#)WizardPageEvent.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.wizard.page.event;

import com.wolfram.guikit.wizard.page.WizardPage;

/**
 * Wizard
 *
 * @version $Revision: 1.2 $
 */
public class WizardPageEvent  extends java.util.EventObject {
    
  private static final long serialVersionUID = -1287987475456484948L;
    
  public WizardPageEvent(WizardPage p) {
    super(p);
    }
    
  public WizardPage getPage() {return (WizardPage)getSource();}
  
}