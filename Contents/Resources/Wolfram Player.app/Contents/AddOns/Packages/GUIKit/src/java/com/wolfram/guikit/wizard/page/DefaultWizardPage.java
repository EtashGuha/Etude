/*
 * @(#)DefaultWizardPage.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.wizard.page;

import com.wolfram.guikit.wizard.Wizard;

/**
 * DefaultWizardPage
 *
 * @version $Revision: 1.1 $
 */
public class DefaultWizardPage extends AbstractWizardPage {
  
  public DefaultWizardPage() {
    super();
    setNavigationMask(Wizard.NAVIGATEBACK | Wizard.NAVIGATENEXT | Wizard.NAVIGATELAST |
      Wizard.NAVIGATECANCEL | Wizard.NAVIGATEHELP);
    }
    
}