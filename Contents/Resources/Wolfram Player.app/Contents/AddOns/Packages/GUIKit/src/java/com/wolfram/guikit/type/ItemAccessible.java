/*
 * @(#)ItemAccessible.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.type;

/**
 * ItemAccessible is an interface that objects can implement to expose
 * their content as arrays of items and an indexed property item.
 */
public interface ItemAccessible {

  public Object getItems();
  public void setItems(Object objs);
  
  public Object getItem(int index);
  public void setItem(int index, Object obj);
  
  }
