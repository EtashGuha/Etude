/*
 * @(#)GUIKitTypedObject.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.type;

import com.wolfram.bsf.util.type.TypedObject;
import com.wolfram.guikit.layout.GUIKitLayout;
import com.wolfram.guikit.layout.GUIKitLayoutFactory;
import com.wolfram.guikit.layout.GUIKitLayoutInfo;

/**
 * GUIKitTypedObject
 */
public class GUIKitTypedObject extends TypedObject {

  public static final GUIKitTypedObject TYPED_NULL = new GUIKitTypedObject(Object.class, null);
  public static final GUIKitTypedObject TYPED_TRUE = new GUIKitTypedObject(boolean.class, Boolean.TRUE);
  public static final GUIKitTypedObject TYPED_FALSE = new GUIKitTypedObject(boolean.class, Boolean.FALSE);
  
  protected GUIKitLayout layout = null;
  protected GUIKitLayoutInfo layoutInfo = null;
  
  protected boolean rootComponent = false;
  protected boolean addedToolbar = false;
  
  public GUIKitTypedObject(Class type, Object value) {
    super(type, value);
    }
    
  public GUIKitLayout getLayout(){return layout;}
  public void setLayout(GUIKitLayout l) {layout = l;}
  
  public GUIKitLayoutInfo getLayoutInfo(){return layoutInfo;}
  public void setLayoutInfo(GUIKitLayoutInfo l) {layoutInfo = l;}
  
  public GUIKitLayout createLayout(GUIKitLayoutInfo layoutInfo) {
    rootComponent = true;
    layout = GUIKitLayoutFactory.createLayout(layoutInfo);
    return layout;
    }

  public boolean getAddedToolbar() {return addedToolbar;}
  public void setAddedToolbar(boolean val) {addedToolbar = val;}
  public boolean isRootComponent() {return rootComponent;}
  public void setRootComponent(boolean r) { rootComponent = r;}

}

