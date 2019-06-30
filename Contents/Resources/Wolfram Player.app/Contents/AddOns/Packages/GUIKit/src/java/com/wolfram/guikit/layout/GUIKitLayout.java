/*
 * @(#)GUIKitLayout.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.layout;

import java.awt.Component;

import javax.swing.JComponent;
import javax.swing.border.Border;

import com.wolfram.guikit.type.GUIKitTypedObject;

/**
 * GUIKitLayout
 */
public interface GUIKitLayout  {

  public static final int GROUP_UNKNOWN = -1;
  public static final int GROUP_NONE = 0;
  public static final int GROUP_AUTOMATIC = 1;
  public static final int GROUP_ROW = 2;
  public static final int GROUP_COLUMN = 3;
  public static final int GROUP_GRID = 4;
	public static final int GROUP_TAB = 5;
	public static final int GROUP_SPLIT = 6;
	
	public static final int ALIGN_UNKNOWN = -1;
  public static final int ALIGN_LEFT = 0;
  public static final int ALIGN_RIGHT = 1;
  public static final int ALIGN_CENTER = 2;
  public static final int ALIGN_TOP = 3;
  public static final int ALIGN_BOTTOM = 4;
  public static final int ALIGN_AUTOMATIC = 5;
  public static final int ALIGN_AFTER = 6;
  public static final int ALIGN_BEFORE = 7;
  
	public static final int STRETCH_UNKNOWN = -1;
  public static final int STRETCH_NONE = 0;
  public static final int STRETCH_COMPONENTALIGN = 1;
  public static final int STRETCH_TRUE = 2;
  public static final int STRETCH_MAXIMIZE = 3;
  
	public static final int TABPLACEMENT_UNKNOWN = -1;
	public static final int TABPLACEMENT_TOP = 0;
	public static final int TABPLACEMENT_BOTTOM = 1;
	public static final int TABPLACEMENT_LEFT = 2;
	public static final int TABPLACEMENT_RIGHT = 3;
	
	public static final int SPLIT_UNKNOWN = -1;
	public static final int SPLIT_VERTICAL = 0;
	public static final int SPLIT_HORIZONTAL = 1;
	
  public boolean pushLayout(GUIKitLayoutInfo layoutInfo);
  
  public void nestBoxAsTab(GUIKitLayoutInfo parentInfo);
  public void nestBoxAsSplit(GUIKitLayoutInfo parentInfo);
  
  public GUIKitLayout createLayout(GUIKitLayoutInfo layoutInfo);
   
  public void popLayout();

	public void applyLayout(GUIKitTypedObject obj);
	
  // What other forms do we expose or wrap with our own static finals
  public void add(Component c);
  public void add(Component c, int xStretching, int yStretching);
  
  public void addBorder(Border b);
  public void addSpace(int n);
  public void addFill();
  
  public void addAlign();
  public void addAlign(Component c, int fromAlign, int toAlign);
  
  public void setInterComponentSpacing(int spacing);
  
  public JComponent getCurrentContainer();
  
  public Component getRoot();
  
  public boolean isEmpty();
  
  public void setAlignment(int alignType);
  public void setAlignment(int horzAlignType, int vertAlignType);

}

