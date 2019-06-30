/*
 * @(#)ItemComboBoxModel.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing;

import java.util.Vector;

import javax.swing.DefaultComboBoxModel;

import com.wolfram.guikit.type.ItemAccessible;

/**
 * ItemComboBoxModel extends DefaultComboBoxModel with a utility
 * "item" indexed get/set property which the default model does not
 * provide
 */
public class ItemComboBoxModel extends DefaultComboBoxModel implements ItemAccessible {

  private static final long serialVersionUID = -1247987975456744949L;
    
  public ItemComboBoxModel() {
    super();
    }
  
  public ItemComboBoxModel(final Object items[]) {
    this();
    setItems(items);
    }
    
  public ItemComboBoxModel(Vector v) {
    this(v.toArray());
    }
     
  public Object getItems() {
    Object[] objs = new Object[getSize()];
    for (int i = 0; i < objs.length; ++i)
      objs[i] = getElementAt(i);
    return objs;
    }
    
  public void setItems(Object objs) {
    removeAllElements();
    Object[] objArray = ItemListModel.convertToArray(objs);
    if (objArray != null) {
      int count = objArray.length;
      for(int i = 0; i < count; ++i)
        addElement(objArray[i]);
      }
    }
    
  public Object getItem(int index) {
    return getElementAt(index);
    }

  public void setItem(int index, Object obj) {
    // Somewhat inefficient but should be using different methods anyway
    Object o = getItems();
    if (o == null || o.getClass() != Object[].class) return;
    Object[] objs = (Object[])o;
    if (index < objs.length) {
      objs[index] = obj;
      setItems(objs);
      }
    }

  }
