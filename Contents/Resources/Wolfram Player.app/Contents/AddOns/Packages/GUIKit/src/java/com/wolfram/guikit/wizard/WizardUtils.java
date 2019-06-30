/*
 * @(#)WizardUtils.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.wizard;

import java.util.Vector;


/**
 * WizardUtils
 *
 * @version $Revision: 1.1 $
 */
public class WizardUtils  {
 
  public static int getNavigationMask(String[] names) {
    int newMask = 0;
    if (names == null) return newMask;
    for(int i = 0; i < names.length; ++i) {
      String name = names[i];
      if (name == null) continue;
      if (name.equalsIgnoreCase("Back")) newMask |= Wizard.NAVIGATEBACK;
      if (name.equalsIgnoreCase("Next")) newMask |= Wizard.NAVIGATENEXT;
      if (name.equalsIgnoreCase("Last")) newMask |= Wizard.NAVIGATELAST;
      if (name.equalsIgnoreCase("Finish")) newMask |= Wizard.NAVIGATEFINISH;
      if (name.equalsIgnoreCase("Cancel")) newMask |= Wizard.NAVIGATECANCEL;
      if (name.equalsIgnoreCase("Close")) newMask |= Wizard.NAVIGATECLOSE;
      if (name.equalsIgnoreCase("Help")) newMask |= Wizard.NAVIGATEHELP;
      }
    return newMask;
    }
    
  public static String[] getNavigationNames(int mask) {
    Vector names = new Vector();
    if ((mask & Wizard.NAVIGATEBACK) != 0) names.add("Back");
    if ((mask & Wizard.NAVIGATENEXT) != 0) names.add("Next");
    if ((mask & Wizard.NAVIGATELAST) != 0) names.add("Last");
    if ((mask & Wizard.NAVIGATEFINISH) != 0) names.add("Finish");
    if ((mask & Wizard.NAVIGATECANCEL) != 0) names.add("Cancel");
    if ((mask & Wizard.NAVIGATECLOSE) != 0) names.add("Close");
    if ((mask & Wizard.NAVIGATEHELP) != 0) names.add("Help");
    return (String[])names.toArray(new String[]{});
    }
    
}