/*
 * @(#)ItemListModel.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing;

import javax.swing.DefaultListModel;

import com.wolfram.bsf.util.MathematicaBSFException;
import com.wolfram.bsf.util.type.MathematicaTypeConvertorRegistry;
import com.wolfram.guikit.type.ItemAccessible;

/**
 * ItemListModel extends DefaultListModel with a utility
 * "item" indexed get/set property which the default model does not
 * provide
 */
public class ItemListModel extends DefaultListModel implements ItemAccessible {

  private static final long serialVersionUID = -1287557975456758948L;
    
  public ItemListModel() {
    super();
    }
  
  public Object getItems() {
    return toArray();
    }
  
  public static Object[] convertToArray(Object objs) {
    Object[] objArray = null;
    if (objs != null) {
      if (objs.getClass() == Object[].class)
        objArray = (Object[])objs;
      else {
        try {
          objArray = (Object[])MathematicaTypeConvertorRegistry.typeConvertorRegistry.convertAsObject(
            objs.getClass(), objs, Object[].class);
          }
        catch (MathematicaBSFException me){}
        }
      }
    return objArray;
    }
    
  public void setItems(Object objs) {
    Object[] objArray = convertToArray(objs);
    if (objArray != null) {
      int count = objArray.length;
      setSize(count);
      for(int i = 0; i < count; ++i)
        setItem(i, objArray[i]);
      }
    else setSize(0);
    }
  
  public Object getItem(int index) {
    return get(index);
    }

  public void setItem(int index, Object obj) {
    set(index, obj);
    }

  }
