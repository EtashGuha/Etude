/*
 * @(#)GUIKitLayoutFactory.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.layout;

import com.wolfram.jlink.JLinkClassLoader;

/**
 * GUIKitLayoutFactory
 */
public class GUIKitLayoutFactory  {

  private static final String DEFAULT_GUIKITLAYOUT_CLASS = "com.wolfram.guikit.layout.DefaultGUIKitLayout";
  
  private static GUIKitLayout layout = null;

	static {
		// We currently instantiate the default layout class using this
		// technique so that if we decided to deploy the OculusLayout based
		// code without source a user could still recompile PACKAGE_CONTEXT
		// sans the DefaultGUIKitLayout.java source
		//layout = new DefaultGUIKitLayout();
		try {
			layout = (GUIKitLayout)JLinkClassLoader.classFromName(DEFAULT_GUIKITLAYOUT_CLASS).newInstance();
		  } 
		catch (Exception e) {}
		if (layout == null) {
			try {
				layout = (GUIKitLayout)Class.forName(DEFAULT_GUIKITLAYOUT_CLASS).newInstance();
				} 
			catch (Exception e) {}
		  }
		}
		
  public static void setLayout(GUIKitLayout l) {layout = l;}

  public static GUIKitLayout createLayout(GUIKitLayoutInfo layoutInfo) {
    if (layout != null)
      return layout.createLayout(layoutInfo);
    else return null;
    }
    
}

