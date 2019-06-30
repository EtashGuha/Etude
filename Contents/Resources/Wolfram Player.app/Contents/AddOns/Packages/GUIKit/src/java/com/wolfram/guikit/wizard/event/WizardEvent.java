/*
 * @(#)Wizard.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.wizard.event;

import com.wolfram.guikit.wizard.Wizard;

/**
 * Wizard
 *
 * @version $Revision: 1.2 $
 */
public class WizardEvent extends java.util.EventObject {
    
  private static final long serialVersionUID = -1287967975456688948L;
    
  public WizardEvent(Wizard w) {
    super(w);
    }
   
  public Wizard getWizard() {return (Wizard)getSource();}
  
}