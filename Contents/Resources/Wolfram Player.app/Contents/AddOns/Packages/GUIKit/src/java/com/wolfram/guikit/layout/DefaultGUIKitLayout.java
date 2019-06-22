/*
 * @(#)DefaultGUIKitLayout.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.layout;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Container;
import java.awt.ScrollPane;

import javax.swing.JComponent;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.JTabbedPane;
import javax.swing.RootPaneContainer;
import javax.swing.border.Border;

import com.oculustech.layout.AlignedComponentSpacing;
import com.oculustech.layout.OculusBox;
import com.oculustech.layout.OculusGrid;
import com.oculustech.layout.OculusLayout;
import com.oculustech.layout.OculusLayoutHelper;
import com.oculustech.layout.OculusLayoutInfo;

import com.wolfram.guikit.type.GUIKitTypedObject;

/**
 * DefaultGUIKitLayout
 */
public class DefaultGUIKitLayout implements GUIKitLayout {

  protected OculusLayoutHelper helper = null;
  protected boolean isEmpty = true;
  
  public DefaultGUIKitLayout() {
    }
  
  public DefaultGUIKitLayout(GUIKitLayoutInfo layoutInfo) {
    if (layoutInfo.getGroupType() == GROUP_ROW)
      helper = new OculusLayoutHelper(OculusLayout.HORIZONTAL);
    else if (layoutInfo.getGroupType() == GROUP_NONE) {
      helper = null;
      }
    else if (layoutInfo.getGroupType() == GROUP_GRID && 
        layoutInfo.getGroupDimensions() != null && layoutInfo.getGroupDimensions().length >= 2) {
      helper = new OculusLayoutHelper();
      helper.nestGrid(layoutInfo.getGroupDimensions()[1], layoutInfo.getGroupDimensions()[0]);
      }
    else if (layoutInfo.getGroupType() == GROUP_TAB) {
      helper = new OculusLayoutHelper();
      int ori = JTabbedPane.TOP;
      if (layoutInfo.getTabPlacement() == GUIKitLayout.TABPLACEMENT_BOTTOM) ori = JTabbedPane.BOTTOM;
      else if (layoutInfo.getTabPlacement() == GUIKitLayout.TABPLACEMENT_RIGHT) ori = JTabbedPane.RIGHT;
      else if (layoutInfo.getTabPlacement() == GUIKitLayout.TABPLACEMENT_LEFT) ori = JTabbedPane.LEFT;
      helper.nestTabbedPane(ori);
      }
    else if (layoutInfo.getGroupType() == GROUP_SPLIT) {
      helper = new OculusLayoutHelper();
      int oriSplit = JSplitPane.VERTICAL_SPLIT;
      if (layoutInfo.getSplitOrientation() == GUIKitLayout.SPLIT_HORIZONTAL) 
        oriSplit = JSplitPane.HORIZONTAL_SPLIT;
      helper.nestSplitPane(oriSplit);
      }
    else
      helper = new OculusLayoutHelper(OculusLayout.VERTICAL);
      
    if (helper != null) {
      // For lowlevel debugging OculusLayout purposes only, 
      // helper.setDebugOutStream(System.err);
      }
    }
	
  public JComponent getCurrentContainer() {
   if (helper != null) return helper.getCurrentContainer();
   else return null;
   }
   
  public boolean isEmpty() {return isEmpty;}
    
  public void nestBoxAsTab(GUIKitLayoutInfo parentInfo) {
    nestBoxAsTab(parentInfo, OculusLayout.VERTICAL);
    }
  
  protected void nestBoxAsTab(GUIKitLayoutInfo parentInfo, int orientation) {
    if (helper != null && parentInfo != null) {
        String tabName = "";
        String[] tabNames = parentInfo.getTabNames();
        int currIndex = parentInfo.getNextChildIndex();
        if (tabNames != null && currIndex < tabNames.length) {
          tabName = tabNames[currIndex];
          }
        helper.nestBoxAsTab(tabName, orientation);
        parentInfo.incrementChildIndex();
      }
    }
  
  public void nestBoxAsSplit(GUIKitLayoutInfo parentInfo) {
    nestBoxAsSplit(parentInfo, OculusLayout.VERTICAL);
    }
    
  protected void nestBoxAsSplit(GUIKitLayoutInfo parentInfo, int orientation) {
    if (helper != null && parentInfo != null) {
        String location = JSplitPane.TOP;
        int currIndex = parentInfo.getNextChildIndex();
        if (parentInfo.getSplitOrientation() == GUIKitLayout.SPLIT_VERTICAL) {
          if (currIndex == 0) location = JSplitPane.TOP;
          else location = JSplitPane.BOTTOM;
          }
        // SPLIT_HORIZONTAL
        else {
          if (currIndex == 0) location = JSplitPane.LEFT;
          else location = JSplitPane.RIGHT;
          }
        helper.nestBoxInSplitPane(location, orientation);
        parentInfo.incrementChildIndex();
      }
    }
    
  protected void nestBox(GUIKitLayoutInfo parentInfo, int orientation) {
    if (helper != null) {
      if (parentInfo == null) {
        helper.nestBox(orientation);
        }
      else if (parentInfo.getGroupType() == GROUP_TAB) {
        nestBoxAsTab(parentInfo, orientation);
        }
      else if (parentInfo.getGroupType() == GROUP_SPLIT) {
        nestBoxAsSplit(parentInfo, orientation);
        }
      else {
        helper.nestBox(orientation);
        }
      
      }
    }
  
  public boolean pushLayout(GUIKitLayoutInfo layoutInfo) {
    boolean shouldPop = false;
    if (helper != null) {
      switch (layoutInfo.groupType) {
        case GROUP_NONE: 
          break;
        case GROUP_ROW: 
          nestBox(layoutInfo.getParent(), OculusLayout.HORIZONTAL);
          shouldPop = true;
          break;
        case GROUP_COLUMN: 
          nestBox(layoutInfo.getParent(), OculusLayout.VERTICAL);
          shouldPop = true;
          break;
        case GROUP_GRID: 
          if (layoutInfo.groupDimensions != null && layoutInfo.groupDimensions.length >= 2) {
            helper.nestGrid(layoutInfo.groupDimensions[1], layoutInfo.groupDimensions[0]);
            shouldPop = true;
            }
          break;
        case GROUP_TAB:
          int ori = JTabbedPane.TOP;
          if (layoutInfo.getTabPlacement() == GUIKitLayout.TABPLACEMENT_BOTTOM) ori = JTabbedPane.BOTTOM;
          else if (layoutInfo.getTabPlacement() == GUIKitLayout.TABPLACEMENT_RIGHT) ori = JTabbedPane.RIGHT;
          else if (layoutInfo.getTabPlacement() == GUIKitLayout.TABPLACEMENT_LEFT) ori = JTabbedPane.LEFT;
          helper.nestTabbedPane(ori);
          // This seems to be needed in some (all?) cases
          shouldPop = true;
          break;
        case GROUP_SPLIT:
          int oriSplit = JSplitPane.VERTICAL_SPLIT;
          if (layoutInfo.getSplitOrientation() == GUIKitLayout.SPLIT_HORIZONTAL) 
            oriSplit = JSplitPane.HORIZONTAL_SPLIT;
          helper.nestSplitPane(oriSplit);
          // This seems to be needed in some (all?) cases
          shouldPop = true;
          break;
        // case GROUP_AUTOMATIC
        default: 
          JComponent curr = helper.getCurrentContainer();
          if (curr == null || curr instanceof JTabbedPane || curr instanceof JSplitPane) {
            // this may never happen??
            nestBox(layoutInfo.getParent(), OculusLayout.VERTICAL);
            shouldPop = true;
            }
          else if (curr instanceof OculusBox) {
            int orient = ((OculusBox)curr).getOrientation();
            if (orient == OculusLayout.VERTICAL)
              nestBox(layoutInfo.getParent(), OculusLayout.HORIZONTAL);
            else
              nestBox(layoutInfo.getParent(), OculusLayout.VERTICAL);
            shouldPop = true;
            }
          else if (curr instanceof OculusGrid) {
            // Inside a level of grid so do not want pop
            }
          break;
        }
      }
    return shouldPop;
    }
  
  public void popLayout() {
    // Do we check for validity in request or would clients check this first?
    // What about tab and split usage here, any differences?
    helper.parent();
    }
  
	public void applyLayout(GUIKitTypedObject obj) {
     //	Is this where we check whether a layout helper was
		 // initiated and set the border layout and root obj?
		 if (obj.isRootComponent()) {
			 // Checks if layout content is not empty before we decide to change the layout
			 // especially useful for simple non-panel widget declarations
			 if (obj.value != null && !obj.getLayout().isEmpty() && (obj.getLayout().getRoot() != null)) {
				 if (obj.value instanceof RootPaneContainer) {
					 Container c = ((RootPaneContainer)obj.value).getContentPane();
           if (c.getLayout() == null || !(c.getLayout() instanceof BorderLayout))
					   c.setLayout(new BorderLayout());
					 c.add( obj.getLayout().getRoot(), BorderLayout.CENTER);
					 obj.setRootComponent(false);
					 }   
				 else if (obj.value instanceof Container) {
					 Container c = (Container)obj.value; 
           if (!(c instanceof ScrollPane || c instanceof JScrollPane)) {
             if (c.getLayout() == null || !(c.getLayout() instanceof BorderLayout))
					     c.setLayout(new BorderLayout());
					   c.add( obj.getLayout().getRoot(), BorderLayout.CENTER);
            }
					 obj.setRootComponent(false);
					 }
				 else {
					 // Do we try and be cute about finding a parent container somehow
					 // or will we be 'guaranteed' that our root objects will be containers??
					 }
				 }
  
			 }		
		
	  }
	
  // What other forms do we expose or wrap with our own static finals
  public void add(Component c) {
    if (helper == null || c == null) return;
    // TODO Decide if this check is required or allows us to turn off other added flags
    if (!helper.getCurrentContainer().isAncestorOf(c)) {
      helper.add(c);
      }
    isEmpty = false;
    }
  
  public void add(Component c, int xStretching, int yStretching) {
    if (helper == null || c == null) return;
    int useStretchX = OculusLayoutInfo.CAN_BE_STRETCHED, useStretchY = OculusLayoutInfo.CAN_BE_STRETCHED;
    switch (xStretching) {
      case GUIKitLayout.STRETCH_NONE: useStretchX = OculusLayoutInfo.NO_STRETCH; break;
      case GUIKitLayout.STRETCH_COMPONENTALIGN: useStretchX = OculusLayoutInfo.STRETCH_ONLY_TO_ALIGN; break;
      case GUIKitLayout.STRETCH_TRUE: useStretchX = OculusLayoutInfo.CAN_BE_STRETCHED; break;
      case GUIKitLayout.STRETCH_MAXIMIZE: useStretchX = OculusLayoutInfo.MAX_STRETCHING_PREFERENCE; break;
      }
    switch (yStretching) {
      case GUIKitLayout.STRETCH_NONE: useStretchY = OculusLayoutInfo.NO_STRETCH; break;
      case GUIKitLayout.STRETCH_COMPONENTALIGN: useStretchY = OculusLayoutInfo.STRETCH_ONLY_TO_ALIGN; break;
      case GUIKitLayout.STRETCH_TRUE: useStretchY = OculusLayoutInfo.CAN_BE_STRETCHED; break;
      case GUIKitLayout.STRETCH_MAXIMIZE: useStretchY = OculusLayoutInfo.MAX_STRETCHING_PREFERENCE; break;
      }
    // TODO Decide if this check is required or allows us to turn off other added flags
    if (!helper.getCurrentContainer().isAncestorOf(c)) {
      helper.add(c, useStretchX, useStretchY);
      }
    isEmpty = false;
    }
  
  public void addBorder(Border b) {
    if (helper == null || b == null) return;
    helper.addBorder(b);
    isEmpty = false;
    }
  
  public void addSpace(int n) {
    if (helper == null) return;
    helper.addSpace(n);
    isEmpty = false;
    }
  
  public void addFill() {
    if (helper == null) return;
    helper.addFiller();
    isEmpty = false;
    }
  
  public void addAlign() {
    if (helper == null) return;
    helper.addAlignmentPoint();
    isEmpty = false;
    }
    
  public void addAlign(Component c, int fromAlign, int toAlign) {
    if (helper == null || c == null) return;
    helper.alignNextComponentTo(c,
       (fromAlign == ALIGN_BEFORE) ? 
            AlignedComponentSpacing.LEADING_EDGE : AlignedComponentSpacing.TRAILING_EDGE,
       (toAlign == ALIGN_AFTER) ? 
            AlignedComponentSpacing.TRAILING_EDGE : AlignedComponentSpacing.LEADING_EDGE   
       );
    isEmpty = false;
    }
  
  public void setInterComponentSpacing(int spacing) {
    if (helper == null) return;
    helper.setInterComponentSpacing(spacing);
    }
  
  public Component getRoot() {
    if (helper == null) return null;
    return helper.getRoot();
    }
  
  
  public void setAlignment(int alignType) {
    if (helper == null) return;
    switch (alignType) {
      case ALIGN_LEFT:  
        helper.setJustification(OculusLayout.JUSTIFY_LEFT);
        break;
      case ALIGN_RIGHT: 
        helper.setJustification(OculusLayout.JUSTIFY_RIGHT);
        break;
      case ALIGN_CENTER: 
        helper.setJustification(OculusLayout.JUSTIFY_CENTER);
        break;
      case ALIGN_TOP: 
        helper.setJustification(OculusLayout.JUSTIFY_TOP);
        break;
      case ALIGN_BOTTOM:        
        helper.setJustification(OculusLayout.JUSTIFY_BOTTOM);
        break;
      // AUTOMATIC mode based on current container
      default: 
        JComponent curr = helper.getCurrentContainer();
        if (curr == null) {
          // this may never happen??
          helper.setJustification(OculusLayout.CENTER);
          }
        else if (curr instanceof OculusBox) {
          int orient = ((OculusBox)curr).getOrientation();
          if (orient == OculusLayout.VERTICAL)
            helper.setJustification(OculusLayout.JUSTIFY_LEFT);
          else
            helper.setJustification(OculusLayout.JUSTIFY_CENTER);
          }
        else if (curr instanceof OculusGrid) {
          // Need to figure out if we are in a column or row level of
          // grid to determine if we add anything
          helper.setGridCellJustification(OculusGrid.DEFAULT_JUSTIFICATION, OculusGrid.DEFAULT_JUSTIFICATION);
          }
        break;
      }
    
    }
  
  public void setAlignment(int horzAlignType, int vertAlignType) {
    if (helper == null) return;
    int horzJustify = OculusGrid.DEFAULT_JUSTIFICATION;
    int vertJustify = OculusGrid.DEFAULT_JUSTIFICATION;
    switch (horzAlignType) {
      case ALIGN_RIGHT: horzJustify = OculusLayout.JUSTIFY_RIGHT; break;
      case ALIGN_CENTER: horzJustify = OculusLayout.JUSTIFY_CENTER; break;
      case ALIGN_LEFT: horzJustify = OculusLayout.JUSTIFY_LEFT; break;
      }
    switch (vertAlignType) {
      case ALIGN_TOP: vertJustify = OculusLayout.JUSTIFY_TOP; break;
      case ALIGN_BOTTOM: vertJustify = OculusLayout.JUSTIFY_BOTTOM; break;
      case ALIGN_CENTER: vertJustify = OculusLayout.JUSTIFY_CENTER; break;
      }
    helper.setGridCellJustification(horzJustify, vertJustify);
    }
  
  static {
    /* Required deployment license for using OculusLayout
       NOTICE: This license is owned by Wolfram Research, Inc.
         It is illegal to use this license outside of this class for use external to
         Mathematica products and the PACKAGE_CONTEXT framework.
       Licenses for OculusLayout are available at http://www.javalayout.com/ 
    */
    OculusLayout.setLicenseNumber("J6A45AM36UX");
    }
 
  public GUIKitLayout createLayout(GUIKitLayoutInfo layoutInfo) {
    return new DefaultGUIKitLayout(layoutInfo);
    }


}

