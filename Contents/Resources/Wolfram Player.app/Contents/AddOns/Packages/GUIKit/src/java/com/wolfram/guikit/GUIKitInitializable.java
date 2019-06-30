/*
 * @(#)GUIKitInitializable.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit;

/**
 * GUIKitInitializable was created as a way for a bean to 
 * hook into the BSF runtime engine as soon as it was created.
 * Currently there are no active implementors of this interface
 * so its initial goals may not be required in the API
 */
public interface GUIKitInitializable  {

	public void guiInit(GUIKitEnvironment env);
	
}
