/*
 * @(#)GUIKitLayoutInfo.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.layout;

import javax.swing.border.Border;

import com.wolfram.guikit.type.GUIKitTypedObject;

/**
 * GUIKitLayoutInfo
 */
public class GUIKitLayoutInfo  {

	protected Border border = null;
	
	protected int groupType = GUIKitLayout.GROUP_UNKNOWN;
	protected int[] groupDimensions = null;
	protected GUIKitTypedObject groupingObject = null;
	
	protected int primaryAlignType = GUIKitLayout.ALIGN_UNKNOWN;
	protected int secondaryAlignType = GUIKitLayout.ALIGN_UNKNOWN;
	
	protected int stretchingX = GUIKitLayout.STRETCH_UNKNOWN;
	protected int stretchingY = GUIKitLayout.STRETCH_UNKNOWN;
	
	protected int spacing = -1;
	
  protected GUIKitLayoutInfo parent = null;
  
  protected int nextChildIndex = 0;
  protected int tabPlacement = GUIKitLayout.TABPLACEMENT_UNKNOWN;
  protected String[] tabNames = null;
  protected int splitOrientation = GUIKitLayout.SPLIT_UNKNOWN;
  protected int[] childOrientations = null;
  protected int[] childJustifications = null;
  
  public GUIKitLayoutInfo() {
  	}
  
  public GUIKitLayoutInfo getParent() {return parent;}
  public void setParent(GUIKitLayoutInfo p) {parent = p;}
  
  public int getGroupType() {return groupType;}
  public void setGroupType(int type) {groupType = type;}
  
  public int[] getGroupDimensions() {return groupDimensions;}
  public void setGroupDimensions(int[] dims) {groupDimensions = dims;}
  
  public GUIKitTypedObject getGroupingObject() {return groupingObject;}
  public void setGroupingObject(GUIKitTypedObject go) {groupingObject = go;}
  
  public void incrementChildIndex() {
    if (groupType == GUIKitLayout.GROUP_TAB || groupType == GUIKitLayout.GROUP_SPLIT)
      setNextChildIndex( getNextChildIndex() + 1);
    else if (getParent() != null)
      getParent().setNextChildIndex( getParent().getNextChildIndex() + 1);
    else setNextChildIndex( getNextChildIndex() + 1);
    }
  public int getNextChildIndex() {
    if (groupType == GUIKitLayout.GROUP_TAB || groupType == GUIKitLayout.GROUP_SPLIT)
      return nextChildIndex;
    else if (getParent() != null)
      return getParent().getNextChildIndex();
    return nextChildIndex;
    }
  public void setNextChildIndex(int i) {
    if (groupType == GUIKitLayout.GROUP_TAB || groupType == GUIKitLayout.GROUP_SPLIT)
      nextChildIndex = i;
    else if (getParent() != null)
      getParent().setNextChildIndex(i);
    else
      nextChildIndex = i;
    }
  public int getTabPlacement() {
    if (groupType == GUIKitLayout.GROUP_TAB)
      return tabPlacement;
    else if (getParent() != null)
      return getParent().getTabPlacement();
    return tabPlacement;
    }
  public void setTabPlacement(int placement) {
    if (groupType == GUIKitLayout.GROUP_TAB)
      tabPlacement = placement;
    else if (getParent() != null)
      getParent().setTabPlacement(placement);
  	else tabPlacement = placement;
  	}
 	public String[] getTabNames() {
    if (groupType == GUIKitLayout.GROUP_TAB)
      return tabNames;
    else if (getParent() != null)
      return getParent().getTabNames();
    return tabNames;
    }
 	public void setTabNames(String[] names) {
    if (groupType == GUIKitLayout.GROUP_TAB)
      tabNames = names;
    else if (getParent() != null)
      getParent().setTabNames(names);
    else tabNames = names;
 		}
	public int getSplitOrientation() {
    if (groupType == GUIKitLayout.GROUP_SPLIT)
      return splitOrientation;
    else if (getParent() != null)
      return getParent().getSplitOrientation();
    return splitOrientation;
    }
	public void setSplitOrientation(int orient) {
    if (groupType == GUIKitLayout.GROUP_SPLIT)
      splitOrientation = orient;
    else if (getParent() != null)
      getParent().setSplitOrientation(orient);
		else splitOrientation = orient;
		}
		
  // Finish these with parent calls or remove completely if not needed
  // If these are the oculus vertical/horizontal these may default to vertical
 	public int[] getChildOrientations() {return childOrientations;}
 	public void setChildOrientations(int[] childOrients) {
		childOrientations = childOrients;
 		}
  // Justifications may also be based on children content?
	public int[] getChildJustifications() {return childJustifications;}
	public void setChildJustifications(int[] childJusts) {
		childJustifications = childJusts;
		}
		
  public Border getBorder() {return border;}
  public void setBorder(Border b) {border = b;}
  
  public int getPrimaryAlignType() {
    if (primaryAlignType != GUIKitLayout.ALIGN_UNKNOWN)
      return primaryAlignType;
    else if (getParent() != null)
      return getParent().getPrimaryAlignType();
    return primaryAlignType;
    }
  public void setPrimaryAlignType(int type) {primaryAlignType = type;}
     
  public int getSecondaryAlignType() {
    if (secondaryAlignType != GUIKitLayout.ALIGN_UNKNOWN)
      return secondaryAlignType;
    else if (getParent() != null)
      return getParent().getSecondaryAlignType();
    return secondaryAlignType;
    }
  public void setSecondaryAlignType(int type) {secondaryAlignType = type;}
  
  public int getSpacing() {
    if (spacing >= 0)
      return spacing;
    else if (getParent() != null)
      return getParent().getSpacing();
    return spacing;
    }
  public void setSpacing(int s) {spacing = s;}
  
  public int getStretchingX() {
    if (stretchingX != GUIKitLayout.STRETCH_UNKNOWN)
      return stretchingX;
    else if (getParent() != null)
      return getParent().getStretchingX();
    return stretchingX;
    }
  public void setStretchingX(int s) {stretchingX = s;}
     
  public int getStretchingY() {
    if (stretchingY != GUIKitLayout.STRETCH_UNKNOWN)
      return stretchingY;
    else if (getParent() != null)
      return getParent().getStretchingY();
    return stretchingY;
    }
  public void setStretchingY(int s) {stretchingY = s;}
  
}

