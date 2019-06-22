/*
 * @(#)WindowUtils.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.util;

import java.awt.*;

/**
 * WindowUtils
 */
public class WindowUtils {

  public static void centerComponent(Component comp) {
    if (comp == null) return;
    
    Dimension screenDim = (comp.getToolkit()).getScreenSize();
    // Should handle case of comp not packed get and width and height are zero
    Dimension compDim   = comp.getSize();

    int width = (screenDim.width  - compDim.width)  / 2;
    int height = (screenDim.height - compDim.height) / 2;

    width  = (width  >= 0 ? width  : 0);
    height = (height >= 0 ? height : 0);
    comp.setLocation(width, height);
    }

  public static void centerComponentIfNeeded(Component comp) {
    if (comp == null) return;
    // NOTE: On Mac OS X comp.getY() returns default non-zero because of
    // global menubar position so we only check for getX on OS X for non default location
    // though this isn't perfect yet, could move to subclass checks but this would not
    // support non GUIKit Window classes
    if (comp.getX() != 0 || (comp.getY() != 0 && !com.wolfram.jlink.Utils.isMacOSX())) return;
    centerComponent(comp);
    }
    
  }